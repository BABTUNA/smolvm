//! Native execution of the host-driven side of the branchpoint handshake.
//!
//! The host used to drive every step — wait, arm, park, release, activate —
//! by building a shell script and running it through `VmExec`, with the marker
//! formats mirrored as `sed` patterns. These functions execute the same steps
//! against the same marker files, so a helper of either vintage sees an
//! identical protocol; what changes is that the rules live once, here, typed.
//!
//! Every function takes its paths, so each is tested against a temp dir the
//! way the scripts were.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use smolvm_protocol::forkpoint::{
    typed_error, ARMED_PREFIX, ARM_PREFIX, GENERATION_PREFIX, LEGACY_RELEASE_TOKEN, RELEASE_PREFIX,
};

/// A failed step, carrying the protocol error code the host handles on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypedError {
    /// One of [`typed_error`].
    pub code: &'static str,
    /// What went wrong, for the operator.
    pub message: String,
}

impl TypedError {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    fn io(step: &str, error: std::io::Error) -> Self {
        Self::new(typed_error::IO, format!("{step}: {error}"))
    }
}

/// Outcome of an activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// This call performed the activation.
    Done,
    /// A previous attempt with the same token had already completed it.
    AlreadyDone,
}

/// Marker files of one clone, all under the forkpoint state directory except
/// the two env files, which live on the workload's filesystem.
#[derive(Debug, Clone)]
pub struct Markers {
    /// The state directory itself.
    pub state_dir: PathBuf,
    /// Written by the helper when the workload reaches its branchpoint.
    pub ready: PathBuf,
    /// Written by the host to enter the restore-safe loop.
    pub arm: PathBuf,
    /// Written by the helper to acknowledge arming.
    pub armed: PathBuf,
    /// Written by the host to let a restored clone continue.
    pub release: PathBuf,
    /// Written by the workload once it is serving (held forks).
    pub worker_ready: PathBuf,
    /// The activation token that claimed this clone.
    pub receipt: PathBuf,
}

impl Markers {
    /// The markers at their protocol locations.
    pub fn standard() -> Self {
        use smolvm_protocol::forkpoint as f;
        Self {
            state_dir: PathBuf::from(f::STATE_DIR),
            ready: PathBuf::from(f::READY_PATH),
            arm: PathBuf::from(f::ARM_PATH),
            armed: PathBuf::from(f::ARMED_PATH),
            release: PathBuf::from(f::RELEASE_PATH),
            worker_ready: PathBuf::from(f::WORKER_READY_PATH),
            receipt: Path::new(f::STATE_DIR).join("activation"),
        }
    }

    /// The same layout rooted at `state_dir`, for tests.
    #[cfg(test)]
    pub fn under(state_dir: &Path) -> Self {
        Self {
            state_dir: state_dir.to_path_buf(),
            ready: state_dir.join("ready"),
            arm: state_dir.join("arm"),
            armed: state_dir.join("armed"),
            release: state_dir.join("release"),
            worker_ready: state_dir.join("worker-ready"),
            receipt: state_dir.join("activation"),
        }
    }
}

/// The scripts' acknowledgement window: 400 polls of 5 ms.
const ACK_POLLS: u32 = 400;
const ACK_INTERVAL: Duration = Duration::from_millis(5);
/// The release script's window: 500 polls of 20 ms.
const RELEASE_POLLS: u32 = 500;
const RELEASE_INTERVAL: Duration = Duration::from_millis(20);

/// Poll `condition` up to `polls` times, `interval` apart; true if it held.
fn poll(polls: u32, interval: Duration, mut condition: impl FnMut() -> bool) -> bool {
    for _ in 0..polls {
        if condition() {
            return true;
        }
        std::thread::sleep(interval);
    }
    condition()
}

/// The generation recorded in a ready marker: the first `generation=` line.
/// `Ok(None)` when the marker carries none or it is not 32 hex digits, which
/// is the legacy helper's marker and is handled by each caller as the script
/// did.
fn generation_of(ready: &Path) -> Result<Option<String>, TypedError> {
    let contents = match std::fs::read_to_string(ready) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(TypedError::new(
                typed_error::NOT_READY,
                "the workload has not declared a branchpoint (no ready marker); it does so by \
                 running `smolvm-branch-ready` once its setup is done",
            ));
        }
        Err(error) => return Err(TypedError::io("read ready marker", error)),
    };
    let generation = contents
        .lines()
        .find_map(|line| line.strip_prefix(GENERATION_PREFIX))
        .unwrap_or("");
    let valid = generation.len() == 32 && generation.bytes().all(|b| b.is_ascii_hexdigit());
    Ok(valid.then(|| generation.to_string()))
}

/// Write `contents` to `path` atomically via a sibling temp file.
fn write_atomic(path: &Path, contents: &str, mode: u32) -> Result<(), TypedError> {
    let tmp = path.with_extension("tmp");
    write_private(&tmp, contents, mode)?;
    std::fs::rename(&tmp, path).map_err(|error| {
        let _ = std::fs::remove_file(&tmp);
        TypedError::io(&format!("move {} into place", path.display()), error)
    })
}

fn write_private(path: &Path, contents: &str, mode: u32) -> Result<(), TypedError> {
    use std::io::Write as _;
    #[cfg(unix)]
    let file = {
        use std::os::unix::fs::OpenOptionsExt as _;
        std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(mode)
            .open(path)
    };
    #[cfg(not(unix))]
    let file = {
        let _ = mode;
        std::fs::File::create(path)
    };
    let mut file =
        file.map_err(|error| TypedError::io(&format!("create {}", path.display()), error))?;
    file.write_all(contents.as_bytes())
        .map_err(|error| TypedError::io(&format!("write {}", path.display()), error))
}

/// Wait until the workload declares its branchpoint; returns the ready
/// marker's contents (its profile lines).
pub fn wait_ready(markers: &Markers, timeout: Duration) -> Result<String, TypedError> {
    let deadline = Instant::now() + timeout;
    loop {
        match std::fs::read_to_string(&markers.ready) {
            Ok(contents) => return Ok(contents),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(TypedError::io("read ready marker", error)),
        }
        if Instant::now() >= deadline {
            return Err(TypedError::new(
                typed_error::NOT_READY,
                format!(
                    "the workload did not reach a branchpoint within {:.1}s; it declares one by \
                     running `smolvm-branch-ready` once its setup is done",
                    timeout.as_secs_f64()
                ),
            ));
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Put a negotiated helper into its restore-safe loop before capture.
pub fn arm(markers: &Markers) -> Result<(), TypedError> {
    let generation = generation_of(&markers.ready)?.ok_or_else(|| {
        TypedError::new(
            typed_error::BAD_GENERATION,
            "the ready marker carries no usable generation; the helper predates arming",
        )
    })?;
    let _ = std::fs::remove_file(&markers.arm);
    if !poll(ACK_POLLS, ACK_INTERVAL, || !markers.armed.exists()) {
        return Err(TypedError::new(
            typed_error::NO_ACK,
            "a previous arming was never cleared by the helper",
        ));
    }
    write_atomic(&markers.arm, &format!("{ARM_PREFIX}{generation}\n"), 0o644)?;
    let expected = format!("{ARMED_PREFIX}{generation}");
    let acknowledged = poll(ACK_POLLS, ACK_INTERVAL, || {
        std::fs::read_to_string(&markers.armed)
            .map(|c| c.lines().any(|line| line == expected))
            .unwrap_or(false)
    });
    if !acknowledged {
        return Err(TypedError::new(
            typed_error::NO_ACK,
            "the helper did not acknowledge arming within the window",
        ));
    }
    Ok(())
}

/// Return a parked source to its ordinary wait after capture.
pub fn park(markers: &Markers) -> Result<(), TypedError> {
    let _ = std::fs::remove_file(&markers.arm);
    if !poll(ACK_POLLS, ACK_INTERVAL, || !markers.armed.exists()) {
        return Err(TypedError::new(
            typed_error::NO_ACK,
            "the helper did not leave its armed loop within the window",
        ));
    }
    Ok(())
}

/// Release a restored clone and wait for the helper to acknowledge.
///
/// The state directory is private guest RAM, so this wakes only this clone
/// even though every clone inherited the same blocked helper.
pub fn release(markers: &Markers) -> Result<(), TypedError> {
    std::fs::create_dir_all(&markers.state_dir)
        .map_err(|error| TypedError::io("create state dir", error))?;
    let generation = match std::fs::metadata(&markers.ready) {
        Ok(_) => generation_of(&markers.ready)?,
        Err(_) => None,
    };
    let marker = match &generation {
        Some(generation) => format!("{RELEASE_PREFIX}{generation}\n"),
        None => format!("{LEGACY_RELEASE_TOKEN}\n"),
    };
    write_atomic(&markers.release, &marker, 0o600)?;
    // The helper acknowledges by dropping its generation line (or the whole
    // marker, for a legacy helper).
    let acknowledged = poll(RELEASE_POLLS, RELEASE_INTERVAL, || match &generation {
        Some(generation) => {
            let line = format!("{GENERATION_PREFIX}{generation}");
            !std::fs::read_to_string(&markers.ready)
                .map(|c| c.lines().any(|l| l == line))
                .unwrap_or(false)
        }
        None => !markers.ready.exists(),
    });
    if !acknowledged {
        return Err(TypedError::new(
            typed_error::NO_ACK,
            "the clone did not acknowledge its release marker",
        ));
    }
    Ok(())
}

/// Assign and release a held clone, idempotently.
///
/// A retry with the same `activation_token` finishes a partial commit; a
/// different token is refused, because a held slot is assigned exactly once.
#[allow(clippy::too_many_arguments)]
pub fn activate(
    markers: &Markers,
    env_dotenv: &str,
    env_sourceable: &str,
    env_path: &Path,
    branch_env_path: &Path,
    require_dir: Option<&Path>,
    env_dir: &Path,
    activation_token: &str,
) -> Result<Activation, TypedError> {
    let receipt_matches = || {
        std::fs::read_to_string(&markers.receipt)
            .map(|c| c.trim_end_matches('\n') == activation_token)
            .unwrap_or(false)
    };
    if markers.release.exists() {
        return if receipt_matches() {
            Ok(Activation::AlreadyDone)
        } else {
            Err(TypedError::new(
                typed_error::TOKEN_MISMATCH,
                "this clone was already activated with a different token",
            ))
        };
    }
    if !markers.ready.exists() {
        return Err(TypedError::new(
            typed_error::NOT_READY,
            "the clone has no ready marker; it was not restored at a branchpoint",
        ));
    }
    let generation = generation_of(&markers.ready)?;
    let _ = std::fs::remove_file(&markers.worker_ready);

    // Claim: a hard link is atomic and fails if the receipt already exists.
    let receipt_tmp = markers.receipt.with_file_name(format!(
        "activation.{activation_token}.{}",
        std::process::id()
    ));
    write_private(&receipt_tmp, &format!("{activation_token}\n"), 0o600)?;
    let claimed = std::fs::hard_link(&receipt_tmp, &markers.receipt).is_ok();
    let _ = std::fs::remove_file(&receipt_tmp);
    if !claimed && !receipt_matches() {
        return Err(TypedError::new(
            typed_error::TOKEN_MISMATCH,
            "another activation claimed this clone first",
        ));
    }

    if let Some(dir) = require_dir {
        if !dir.is_dir() {
            return Err(TypedError::new(
                typed_error::IO,
                format!("missing {}", dir.display()),
            ));
        }
    }
    std::fs::create_dir_all(env_dir)
        .map_err(|error| TypedError::io("create env directory", error))?;
    write_atomic(env_path, env_dotenv, 0o600)?;
    write_atomic(branch_env_path, env_sourceable, 0o600)?;
    let marker = match generation {
        Some(generation) => format!("{RELEASE_PREFIX}{generation}\n"),
        None => format!("{LEGACY_RELEASE_TOKEN}\n"),
    };
    write_atomic(&markers.release, &marker, 0o600)?;
    Ok(Activation::Done)
}

/// Wait until the released workload publishes `token` as worker-ready.
pub fn wait_worker_ready(
    markers: &Markers,
    token: &str,
    timeout: Duration,
) -> Result<(), TypedError> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(contents) = std::fs::read_to_string(&markers.worker_ready) {
            return if contents.trim_end_matches('\n') == token {
                Ok(())
            } else {
                Err(TypedError::new(
                    typed_error::TOKEN_MISMATCH,
                    "the workload published a different worker-ready token",
                ))
            };
        }
        if Instant::now() >= deadline {
            return Err(TypedError::new(
                typed_error::NO_ACK,
                format!(
                    "the workload did not publish worker-ready within {:.1}s",
                    timeout.as_secs_f64()
                ),
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ready_with(markers: &Markers, generation: &str) {
        std::fs::create_dir_all(&markers.state_dir).unwrap();
        std::fs::write(
            &markers.ready,
            format!("smolvm-forkpoint-v1\n{GENERATION_PREFIX}{generation}\n"),
        )
        .unwrap();
    }

    const GEN: &str = "0123456789abcdef0123456789abcdef";

    #[test]
    fn wait_ready_returns_the_marker_or_times_out_with_guidance() {
        let temp = tempfile::tempdir().unwrap();
        let markers = Markers::under(&temp.path().join("state"));
        let err = wait_ready(&markers, Duration::from_millis(120)).unwrap_err();
        assert_eq!(err.code, typed_error::NOT_READY);
        assert!(err.message.contains("smolvm-branch-ready"));
        ready_with(&markers, GEN);
        assert!(wait_ready(&markers, Duration::from_secs(1))
            .unwrap()
            .contains(GEN));
    }

    #[test]
    fn arm_requires_a_generation_and_waits_for_the_ack() {
        let temp = tempfile::tempdir().unwrap();
        let markers = Markers::under(&temp.path().join("state"));
        std::fs::create_dir_all(&markers.state_dir).unwrap();
        std::fs::write(&markers.ready, "smolvm-forkpoint-v1\n").unwrap();
        assert_eq!(arm(&markers).unwrap_err().code, typed_error::BAD_GENERATION);
        ready_with(&markers, GEN);
        // A helper answers arming by publishing the armed marker.
        let armed = markers.armed.clone();
        let arm_path = markers.arm.clone();
        let helper = std::thread::spawn(move || {
            for _ in 0..400 {
                if let Ok(c) = std::fs::read_to_string(&arm_path) {
                    let gen = c.trim_start_matches(ARM_PREFIX).trim();
                    std::fs::write(&armed, format!("{ARMED_PREFIX}{gen}\n")).unwrap();
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        arm(&markers).unwrap();
        helper.join().unwrap();
        assert_eq!(
            std::fs::read_to_string(&markers.arm).unwrap(),
            format!("{ARM_PREFIX}{GEN}\n")
        );
        // park clears arm and waits for armed to go.
        let armed = markers.armed.clone();
        let helper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            std::fs::remove_file(&armed).unwrap();
        });
        park(&markers).unwrap();
        helper.join().unwrap();
        assert!(!markers.arm.exists());
    }

    #[test]
    fn arm_reports_a_silent_helper() {
        let temp = tempfile::tempdir().unwrap();
        let markers = Markers::under(&temp.path().join("state"));
        ready_with(&markers, GEN);
        assert_eq!(arm(&markers).unwrap_err().code, typed_error::NO_ACK);
    }

    #[test]
    fn release_writes_the_generation_marker_and_waits_for_the_ack() {
        let temp = tempfile::tempdir().unwrap();
        let markers = Markers::under(&temp.path().join("state"));
        ready_with(&markers, GEN);
        let ready = markers.ready.clone();
        let release_path = markers.release.clone();
        let helper = std::thread::spawn(move || {
            for _ in 0..500 {
                if let Ok(c) = std::fs::read_to_string(&release_path) {
                    assert_eq!(c, format!("{RELEASE_PREFIX}{GEN}\n"));
                    std::fs::write(&ready, "smolvm-forkpoint-v1\n").unwrap(); // ack: drop the line
                    return;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        });
        release(&markers).unwrap();
        helper.join().unwrap();
        // A legacy helper (no generation) gets the legacy token and acks by removing ready.
        std::fs::write(&markers.ready, "ready\n").unwrap();
        let ready = markers.ready.clone();
        let helper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            std::fs::remove_file(&ready).unwrap();
        });
        release(&markers).unwrap();
        helper.join().unwrap();
        assert_eq!(
            std::fs::read_to_string(&markers.release).unwrap(),
            format!("{LEGACY_RELEASE_TOKEN}\n")
        );
    }

    #[test]
    fn activate_is_idempotent_per_token_and_refuses_another() {
        let temp = tempfile::tempdir().unwrap();
        let markers = Markers::under(&temp.path().join("state"));
        ready_with(&markers, GEN);
        std::fs::write(&markers.worker_ready, "stale\n").unwrap();
        let ws = temp.path().join("workspace");
        let env = ws.join("fork-env");
        let benv = ws.join("branch-env");
        let first = activate(
            &markers,
            "LR=3e-4\n",
            "export LR='3e-4'\n",
            &env,
            &benv,
            None,
            &ws,
            "tok-a",
        )
        .unwrap();
        assert_eq!(first, Activation::Done);
        assert_eq!(std::fs::read_to_string(&env).unwrap(), "LR=3e-4\n");
        assert_eq!(
            std::fs::read_to_string(&benv).unwrap(),
            "export LR='3e-4'\n"
        );
        assert_eq!(
            std::fs::read_to_string(&markers.release).unwrap(),
            format!("{RELEASE_PREFIX}{GEN}\n")
        );
        assert!(!markers.worker_ready.exists());
        // Retry with the same token: already done, nothing rewritten.
        let again = activate(
            &markers,
            "LR=changed\n",
            "export LR='changed'\n",
            &env,
            &benv,
            None,
            &ws,
            "tok-a",
        )
        .unwrap();
        assert_eq!(again, Activation::AlreadyDone);
        assert_eq!(std::fs::read_to_string(&env).unwrap(), "LR=3e-4\n");
        // A different token is refused.
        let other = activate(&markers, "LR=x\n", "", &env, &benv, None, &ws, "tok-b").unwrap_err();
        assert_eq!(other.code, typed_error::TOKEN_MISMATCH);
    }

    #[test]
    fn activate_refuses_a_clone_that_is_not_at_a_branchpoint() {
        let temp = tempfile::tempdir().unwrap();
        let markers = Markers::under(&temp.path().join("state"));
        std::fs::create_dir_all(&markers.state_dir).unwrap();
        let ws = temp.path().join("ws");
        let err = activate(
            &markers,
            "",
            "",
            &ws.join("e"),
            &ws.join("b"),
            None,
            &ws,
            "t",
        )
        .unwrap_err();
        assert_eq!(err.code, typed_error::NOT_READY);
    }

    #[test]
    fn activate_refuses_to_fabricate_a_missing_overlay_root() {
        let temp = tempfile::tempdir().unwrap();
        let markers = Markers::under(&temp.path().join("state"));
        std::fs::create_dir_all(&markers.state_dir).unwrap();
        std::fs::write(&markers.ready, format!("ready\ngeneration={GEN}\n")).unwrap();
        let merged = temp.path().join("merged");
        let env_dir = merged.join("etc/smolvm");
        let err = activate(
            &markers,
            "",
            "",
            &env_dir.join("fork-env"),
            &env_dir.join("branch-env"),
            Some(&merged),
            &env_dir,
            "tok",
        )
        .unwrap_err();
        assert_eq!(err.code, typed_error::IO);
        assert!(!merged.exists());
        assert!(!markers.release.exists());
    }

    #[test]
    fn worker_ready_matches_the_token_or_reports_why() {
        let temp = tempfile::tempdir().unwrap();
        let markers = Markers::under(&temp.path().join("state"));
        std::fs::create_dir_all(&markers.state_dir).unwrap();
        let err = wait_worker_ready(&markers, "t", Duration::from_millis(150)).unwrap_err();
        assert_eq!(err.code, typed_error::NO_ACK);
        std::fs::write(&markers.worker_ready, "other\n").unwrap();
        assert_eq!(
            wait_worker_ready(&markers, "t", Duration::from_secs(1))
                .unwrap_err()
                .code,
            typed_error::TOKEN_MISMATCH
        );
        std::fs::write(&markers.worker_ready, "t\n").unwrap();
        wait_worker_ready(&markers, "t", Duration::from_secs(1)).unwrap();
    }
}
