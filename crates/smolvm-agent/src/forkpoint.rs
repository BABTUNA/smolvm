//! Workload-facing live-branch coordination.
//!
//! A workload calls `smolvm-branch-ready` after initialization. The helper writes
//! a marker and blocks, so the application cannot mutate training state while
//! the host captures the source. Restored children are released independently by
//! the host through a VM-private directory bind-mounted into the container.

use std::io::{Read as _, Write as _};
use std::os::unix::fs::PermissionsExt as _;
use std::path::Path;
use std::time::Duration;

#[cfg(target_os = "linux")]
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
#[cfg(target_os = "linux")]
use std::os::unix::ffi::OsStrExt as _;

const AGENT_BINARY: &str = "/usr/local/bin/smolvm-agent";
use smolvm_protocol::forkpoint::{
    ARMED_PATH, ARMED_PREFIX, ARM_PATH, ARM_PREFIX, BRANCH_ENV_PATH, BRANCH_HELPER_PATH,
    CUDA_PRELOAD_MODULES_HINT, FORK_ENV_PATH, GENERATION_PREFIX, HELPER_PATH, LEGACY_RELEASE_TOKEN,
    READY_PATH, READY_VERSION, RELEASE_PATH, RELEASE_PREFIX, RESTORED_CONTAINER_PATH,
    RESTORED_PATH, STATE_DIR, WORKER_READY_HELPER_PATH, WORKER_READY_PATH, WORKER_READY_TOKEN_ENV,
};

fn enabled() -> bool {
    std::env::var(smolvm_protocol::guest_env::FORKABLE).as_deref()
        == Ok(smolvm_protocol::guest_env::VALUE_ON)
}

/// Whether the paired host supports parking and arming an idle branchpoint.
///
/// This is negotiated by the host at boot. A new guest launched by an older
/// host leaves it unset and retains the legacy always-spinning behavior.
pub fn arming_enabled() -> bool {
    std::env::var(smolvm_protocol::guest_env::BRANCHPOINT_ARMING).as_deref()
        == Ok(smolvm_protocol::guest_env::VALUE_ON)
}

/// Whether this process invocation is the workload-facing helper rather than
/// the PID-1 guest agent.
pub fn helper_requested() -> bool {
    let mut args = std::env::args_os();
    let argv0 = args.next().unwrap_or_default();
    let helper_argv0 = Path::new(&argv0)
        .file_name()
        .is_some_and(|name| name == "smolvm-fork-ready" || name == "smolvm-branch-ready");
    helper_argv0
        || args
            .next()
            .is_some_and(|arg| arg == "fork-ready" || arg == "branch-ready")
}

/// Whether this invocation is the post-restore worker-readiness helper.
pub fn worker_ready_helper_requested() -> bool {
    let mut args = std::env::args_os();
    let argv0 = args.next().unwrap_or_default();
    let helper_argv0 = Path::new(&argv0)
        .file_name()
        .is_some_and(|name| name == "smolvm-worker-ready");
    helper_argv0 || args.next().is_some_and(|arg| arg == "worker-ready")
}

/// Prepare the VM-private coordination directory and the bare-VM helper name.
/// Container workloads receive the same directory and binary through OCI bind
/// mounts in [`inject_into_container`].
pub fn setup() {
    if !enabled() {
        return;
    }
    if let Err(error) = std::fs::create_dir_all(STATE_DIR) {
        tracing::warn!(%error, "failed to create forkpoint state directory");
        return;
    }
    if let Err(error) = std::fs::set_permissions(STATE_DIR, std::fs::Permissions::from_mode(0o1777))
    {
        tracing::warn!(%error, "failed to set forkpoint state permissions");
    }
    let _ = std::fs::remove_file(READY_PATH);
    let _ = std::fs::remove_file(RESTORED_CONTAINER_PATH);
    let _ = std::fs::remove_file(RESTORED_PATH);
    let _ = std::fs::remove_file(RELEASE_PATH);
    let _ = std::fs::remove_file(WORKER_READY_PATH);
    let _ = std::fs::remove_file(ARM_PATH);
    let _ = std::fs::remove_file(ARMED_PATH);

    for helper in [BRANCH_HELPER_PATH, HELPER_PATH, WORKER_READY_HELPER_PATH] {
        if !Path::new(helper).exists() {
            if let Err(error) = std::os::unix::fs::symlink(AGENT_BINARY, helper) {
                tracing::warn!(%error, helper, "failed to install bare-VM forkpoint helper");
            }
        }
    }
}

/// Expose the forkpoint helper and its VM-private state directory inside a
/// workload container. No-op for ordinary machines.
pub fn inject_into_container(spec: &mut crate::oci::OciSpec) {
    inject_into_container_if(spec, enabled(), arming_enabled(), AGENT_BINARY, STATE_DIR);
}

/// Keep a `crun exec` process on the same negotiated arming protocol as the
/// container it joins.
///
/// `crun exec --env` builds a fresh process environment instead of inheriting
/// the container spec. Without restoring this internal variable, a
/// `smolvm-branch-ready` helper launched through `machine exec` falls back to
/// the legacy spin loop while the host waits for an arming acknowledgement.
pub fn augment_exec_env(mut env: Vec<(String, String)>) -> Vec<(String, String)> {
    augment_exec_env_if(&mut env, arming_enabled());
    env
}

fn augment_exec_env_if(env: &mut Vec<(String, String)>, armable: bool) {
    let key = smolvm_protocol::guest_env::BRANCHPOINT_ARMING;
    env.retain(|(existing, _)| existing != key);
    if armable {
        env.push((
            key.to_string(),
            smolvm_protocol::guest_env::VALUE_ON.to_string(),
        ));
    }
}

fn inject_into_container_if(
    spec: &mut crate::oci::OciSpec,
    enabled: bool,
    armable: bool,
    agent_binary: &str,
    state_dir: &str,
) {
    if !enabled || !Path::new(agent_binary).is_file() || !Path::new(state_dir).is_dir() {
        return;
    }
    spec.add_bind_mount(agent_binary, HELPER_PATH, true);
    spec.add_bind_mount(agent_binary, BRANCH_HELPER_PATH, true);
    spec.add_bind_mount(agent_binary, WORKER_READY_HELPER_PATH, true);
    spec.add_bind_mount(state_dir, STATE_DIR, false);
    if armable {
        spec.add_env(
            smolvm_protocol::guest_env::BRANCHPOINT_ARMING,
            smolvm_protocol::guest_env::VALUE_ON,
        );
    }
}

/// Mark the workload ready and block until this VM is a released clone.
pub fn run_helper() -> i32 {
    let preload_modules = std::env::args_os().any(|argument| argument == "--cuda-preload-modules");
    if let Err(error) = run_helper_inner(preload_modules) {
        eprintln!("smolvm-branch-ready: {error}");
        return 1;
    }
    0
}

fn run_helper_inner(preload_modules: bool) -> Result<(), String> {
    run_helper_at(
        ForkpointPaths {
            state_dir: Path::new(STATE_DIR),
            ready_path: Path::new(READY_PATH),
            restored_path: Path::new(RESTORED_PATH),
            release_path: Path::new(RELEASE_PATH),
            arm_path: Path::new(ARM_PATH),
            armed_path: Path::new(ARMED_PATH),
        },
        Duration::from_millis(20),
        preload_modules,
        arming_enabled(),
    )
}

/// Block in the guest kernel until the host changes branchpoint state. The
/// helper always checks the marker files before waiting, so the inotify queue
/// closes the check/sleep race without periodic vCPU wakeups. If inotify is
/// unavailable, the caller retains the bounded polling fallback.
#[cfg(target_os = "linux")]
struct StateChangeWaiter {
    fd: OwnedFd,
}

#[cfg(target_os = "linux")]
impl StateChangeWaiter {
    fn new(state_dir: &Path) -> std::io::Result<Self> {
        let path = std::ffi::CString::new(state_dir.as_os_str().as_bytes()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "branchpoint state path contains NUL",
            )
        })?;
        let raw_fd = unsafe { libc::inotify_init1(libc::IN_CLOEXEC | libc::IN_NONBLOCK) };
        if raw_fd < 0 {
            return Err(std::io::Error::last_os_error());
        }
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };
        let mask = libc::IN_CREATE
            | libc::IN_DELETE
            | libc::IN_MOVED_FROM
            | libc::IN_MOVED_TO
            | libc::IN_CLOSE_WRITE;
        if unsafe { libc::inotify_add_watch(fd.as_raw_fd(), path.as_ptr(), mask) } < 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self { fd })
    }

    fn wait(&self) -> bool {
        let mut descriptor = libc::pollfd {
            fd: self.fd.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        loop {
            let ready = unsafe { libc::poll(&mut descriptor, 1, -1) };
            if ready > 0 {
                let mut events = [0_u8; 1024];
                let _ = unsafe {
                    libc::read(
                        self.fd.as_raw_fd(),
                        events.as_mut_ptr().cast(),
                        events.len(),
                    )
                };
                return true;
            }
            if ready == 0 {
                continue;
            }
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            return false;
        }
    }
}

struct ForkpointPaths<'a> {
    state_dir: &'a Path,
    ready_path: &'a Path,
    restored_path: &'a Path,
    release_path: &'a Path,
    arm_path: &'a Path,
    armed_path: &'a Path,
}

fn run_helper_at(
    paths: ForkpointPaths<'_>,
    poll_interval: Duration,
    preload_modules: bool,
    use_arming: bool,
) -> Result<(), String> {
    let ForkpointPaths {
        state_dir,
        ready_path,
        restored_path,
        release_path,
        arm_path,
        armed_path,
    } = paths;
    std::fs::create_dir_all(state_dir)
        .map_err(|error| format!("create {}: {error}", state_dir.display()))?;
    let _ = std::fs::remove_file(release_path);

    let generation = forkpoint_generation()?;
    let ready_temp = state_dir.join(format!(".ready.{}.tmp", std::process::id()));
    let mut ready = std::fs::File::create(&ready_temp)
        .map_err(|error| format!("create {}: {error}", ready_temp.display()))?;
    let ready_content = ready_content(preload_modules, &generation);
    ready
        .write_all(ready_content.as_bytes())
        .map_err(|error| format!("write {}: {error}", ready_temp.display()))?;
    ready
        .sync_all()
        .map_err(|error| format!("sync {}: {error}", ready_temp.display()))?;
    std::fs::rename(&ready_temp, ready_path).map_err(|error| {
        let _ = std::fs::remove_file(&ready_temp);
        format!(
            "publish {} as {}: {error}",
            ready_temp.display(),
            ready_path.display()
        )
    })?;
    println!("smolvm branch point ready; waiting for child release");
    let _ = std::io::stdout().flush();

    #[cfg(target_os = "linux")]
    let mut change_waiter = use_arming
        .then(|| StateChangeWaiter::new(state_dir).ok())
        .flatten();

    // A timed kernel wait captured in the snapshot is not reliably re-armed by
    // every VMM restore path. Older hosts therefore require the legacy
    // userspace loop below. A paired host advertises the arming protocol: the
    // source sleeps while idle, the host writes ARM_PATH immediately before
    // capture and waits for ARMED_PATH, and only that short capture window uses
    // the restore-safe userspace loop. Once the source continues, removing the
    // arm marker parks it again instead of consuming one host core forever.
    if use_arming {
        while !restored_path.is_file() {
            if release_matches(release_path, &generation) {
                acknowledge_generation(ready_path, &generation);
                return Ok(());
            }
            if arm_matches(arm_path, &generation) {
                publish_generation_marker(armed_path, ARMED_PREFIX, &generation)?;
                while !restored_path.is_file() && arm_matches(arm_path, &generation) {
                    if release_matches(release_path, &generation) {
                        let _ = std::fs::remove_file(armed_path);
                        acknowledge_generation(ready_path, &generation);
                        return Ok(());
                    }
                    std::thread::yield_now();
                }
                let _ = std::fs::remove_file(armed_path);
            } else {
                #[cfg(target_os = "linux")]
                if let Some(waiter) = &change_waiter {
                    if waiter.wait() {
                        continue;
                    }
                    change_waiter = None;
                }
                std::thread::sleep(poll_interval);
            }
        }
    } else {
        while !restored_path.is_file() {
            if release_matches(release_path, &generation) {
                acknowledge_generation(ready_path, &generation);
                return Ok(());
            }
            std::thread::yield_now();
        }
    }

    loop {
        if release_matches(release_path, &generation) {
            acknowledge_generation(ready_path, &generation);
            return Ok(());
        }
        std::thread::sleep(poll_interval);
    }
}

/// Publish the host-issued activation token after clone-local setup completes.
pub fn run_worker_ready_helper() -> i32 {
    let env_path = if Path::new(BRANCH_ENV_PATH).is_file() {
        BRANCH_ENV_PATH
    } else {
        FORK_ENV_PATH
    };
    if let Err(error) = write_worker_ready_at(
        Path::new(STATE_DIR),
        Path::new(env_path),
        Path::new(WORKER_READY_PATH),
    ) {
        eprintln!("smolvm-worker-ready: {error}");
        return 1;
    }
    0
}

fn worker_ready_token(env_path: &Path) -> Result<String, String> {
    let contents = std::fs::read_to_string(env_path)
        .map_err(|error| format!("read {}: {error}", env_path.display()))?;
    let prefix = format!("{WORKER_READY_TOKEN_ENV}=");
    let mut matches = contents
        .lines()
        .filter_map(|line| line.strip_prefix(&prefix));
    let token = matches
        .next()
        .ok_or_else(|| format!("{WORKER_READY_TOKEN_ENV} is not configured for this lease"))?;
    if matches.next().is_some() {
        return Err(format!("{WORKER_READY_TOKEN_ENV} is duplicated"));
    }
    if token.len() != 64 || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!(
            "{WORKER_READY_TOKEN_ENV} must be 64 hexadecimal characters"
        ));
    }
    Ok(token.to_ascii_lowercase())
}

fn write_worker_ready_at(
    state_dir: &Path,
    env_path: &Path,
    worker_ready_path: &Path,
) -> Result<(), String> {
    std::fs::create_dir_all(state_dir)
        .map_err(|error| format!("create {}: {error}", state_dir.display()))?;
    let token = worker_ready_token(env_path)?;
    let temporary = state_dir.join(format!(".worker-ready.{}", std::process::id()));
    let _ = std::fs::remove_file(&temporary);
    let mut marker = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("create {}: {error}", temporary.display()))?;
    if let Err(error) = marker
        .write_all(format!("{token}\n").as_bytes())
        .and_then(|()| marker.sync_all())
    {
        let _ = std::fs::remove_file(&temporary);
        return Err(format!("write {}: {error}", temporary.display()));
    }
    std::fs::rename(&temporary, worker_ready_path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        format!("publish {}: {error}", worker_ready_path.display())
    })
}

fn forkpoint_generation() -> Result<String, String> {
    let mut bytes = [0_u8; 16];
    std::fs::File::open("/dev/urandom")
        .and_then(|mut random| random.read_exact(&mut bytes))
        .map_err(|error| format!("generate forkpoint token: {error}"))?;
    Ok(bytes.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn ready_content(preload_modules: bool, generation: &str) -> String {
    if preload_modules {
        format!("{READY_VERSION}\n{GENERATION_PREFIX}{generation}\n{CUDA_PRELOAD_MODULES_HINT}\n")
    } else {
        format!("{READY_VERSION}\n{GENERATION_PREFIX}{generation}\n")
    }
}

fn release_matches(release_path: &Path, generation: &str) -> bool {
    std::fs::read_to_string(release_path).is_ok_and(|release| {
        let release = release.trim();
        release == format!("{RELEASE_PREFIX}{generation}") || release == LEGACY_RELEASE_TOKEN
    })
}

fn arm_matches(arm_path: &Path, generation: &str) -> bool {
    marker_matches(arm_path, ARM_PREFIX, generation)
}

fn marker_matches(path: &Path, prefix: &str, generation: &str) -> bool {
    std::fs::read_to_string(path)
        .is_ok_and(|marker| marker.trim() == format!("{prefix}{generation}"))
}

fn publish_generation_marker(path: &Path, prefix: &str, generation: &str) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent", path.display()))?;
    let temporary = parent.join(format!(".armed.{}.tmp", std::process::id()));
    let mut marker = std::fs::File::create(&temporary)
        .map_err(|error| format!("create {}: {error}", temporary.display()))?;
    if let Err(error) = marker
        .write_all(format!("{prefix}{generation}\n").as_bytes())
        .and_then(|()| marker.sync_all())
    {
        let _ = std::fs::remove_file(&temporary);
        return Err(format!("write {}: {error}", temporary.display()));
    }
    std::fs::rename(&temporary, path).map_err(|error| {
        let _ = std::fs::remove_file(&temporary);
        format!("publish {}: {error}", path.display())
    })
}

fn acknowledge_generation(ready_path: &Path, generation: &str) {
    let is_current = std::fs::read_to_string(ready_path).is_ok_and(|ready| {
        ready
            .lines()
            .any(|line| line == format!("{GENERATION_PREFIX}{generation}"))
    });
    if is_current {
        let _ = std::fs::remove_file(ready_path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oci::{OciSpec, ProcessIdentity};

    fn wait_for_marker(path: &Path) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !path.is_file() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(
            path.is_file(),
            "marker was not published: {}",
            path.display()
        );
    }

    fn marker_generation(path: &Path) -> String {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .find_map(|line| line.strip_prefix(GENERATION_PREFIX))
            .unwrap()
            .to_string()
    }

    fn spec() -> OciSpec {
        OciSpec::new(
            &["true".to_string()],
            &[],
            "/",
            false,
            &ProcessIdentity::root(),
            false,
        )
    }

    #[test]
    fn ordinary_container_does_not_receive_helper() {
        let mut spec = spec();
        inject_into_container_if(&mut spec, false, false, "/missing-agent", "/missing-state");
        assert!(spec
            .mounts
            .iter()
            .all(|mount| mount.destination != HELPER_PATH
                && mount.destination != BRANCH_HELPER_PATH
                && mount.destination != WORKER_READY_HELPER_PATH));
    }

    #[test]
    fn forkable_container_mounts_helper_and_state() {
        let temp = tempfile::tempdir().unwrap();
        let agent = temp.path().join("smolvm-agent");
        let state = temp.path().join("forkpoint");
        std::fs::write(&agent, b"agent").unwrap();
        std::fs::set_permissions(&agent, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::create_dir(&state).unwrap();
        let mut spec = spec();
        inject_into_container_if(
            &mut spec,
            true,
            true,
            agent.to_str().unwrap(),
            state.to_str().unwrap(),
        );
        let helper = spec
            .mounts
            .iter()
            .find(|mount| mount.destination == HELPER_PATH)
            .unwrap();
        assert_eq!(helper.source, agent.to_str().unwrap());
        assert!(helper.options.iter().any(|option| option == "ro"));
        let branch_helper = spec
            .mounts
            .iter()
            .find(|mount| mount.destination == BRANCH_HELPER_PATH)
            .unwrap();
        assert_eq!(branch_helper.source, agent.to_str().unwrap());
        assert!(branch_helper.options.iter().any(|option| option == "ro"));
        let worker_ready_helper = spec
            .mounts
            .iter()
            .find(|mount| mount.destination == WORKER_READY_HELPER_PATH)
            .unwrap();
        assert_eq!(worker_ready_helper.source, agent.to_str().unwrap());
        assert!(worker_ready_helper
            .options
            .iter()
            .any(|option| option == "ro"));
        assert!(spec.process.env.contains(&format!(
            "{}={}",
            smolvm_protocol::guest_env::BRANCHPOINT_ARMING,
            smolvm_protocol::guest_env::VALUE_ON
        )));
        let state_mount = spec
            .mounts
            .iter()
            .find(|mount| mount.destination == STATE_DIR)
            .unwrap();
        assert_eq!(state_mount.source, state.to_str().unwrap());
        assert!(!state_mount.options.iter().any(|option| option == "ro"));
    }

    #[test]
    fn older_host_does_not_enable_arming_in_a_new_container() {
        let temp = tempfile::tempdir().unwrap();
        let agent = temp.path().join("smolvm-agent");
        let state = temp.path().join("forkpoint");
        std::fs::write(&agent, b"agent").unwrap();
        std::fs::set_permissions(&agent, std::fs::Permissions::from_mode(0o755)).unwrap();
        std::fs::create_dir(&state).unwrap();
        let mut spec = spec();

        inject_into_container_if(
            &mut spec,
            true,
            false,
            agent.to_str().unwrap(),
            state.to_str().unwrap(),
        );

        let prefix = format!("{}=", smolvm_protocol::guest_env::BRANCHPOINT_ARMING);
        assert!(spec
            .process
            .env
            .iter()
            .all(|entry| !entry.starts_with(&prefix)));
    }

    #[test]
    fn exec_env_matches_the_negotiated_arming_protocol() {
        let key = smolvm_protocol::guest_env::BRANCHPOINT_ARMING;
        let mut env = vec![
            ("USER_VALUE".to_string(), "kept".to_string()),
            (key.to_string(), "stale".to_string()),
        ];

        augment_exec_env_if(&mut env, true);
        assert!(env.contains(&("USER_VALUE".to_string(), "kept".to_string())));
        assert_eq!(
            env.iter().filter(|(existing, _)| existing == key).count(),
            1
        );
        assert!(env.contains(&(
            key.to_string(),
            smolvm_protocol::guest_env::VALUE_ON.to_string()
        )));

        augment_exec_env_if(&mut env, false);
        assert!(env.iter().all(|(existing, _)| existing != key));
    }

    #[test]
    fn helper_blocks_until_release_marker() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("forkpoint");
        let ready = state.join("ready");
        let restored = state.join("restored");
        let release = state.join("release");
        let arm = state.join("arm");
        let armed = state.join("armed");
        std::fs::create_dir(&state).unwrap();

        let state_thread = state.clone();
        let ready_thread = ready.clone();
        let release_thread = release.clone();
        let helper = std::thread::spawn(move || {
            run_helper_at(
                ForkpointPaths {
                    state_dir: &state_thread,
                    ready_path: &ready_thread,
                    restored_path: &state_thread.join("restored"),
                    release_path: &release_thread,
                    arm_path: &state_thread.join("arm"),
                    armed_path: &state_thread.join("armed"),
                },
                Duration::from_millis(1),
                false,
                false,
            )
        });
        wait_for_marker(&ready);
        let generation = marker_generation(&ready);
        assert_eq!(
            std::fs::read_to_string(&ready).unwrap(),
            ready_content(false, &generation)
        );
        assert!(!helper.is_finished());
        std::fs::write(&restored, b"restored\n").unwrap();
        std::fs::write(&release, format!("{RELEASE_PREFIX}{generation}\n")).unwrap();
        helper.join().unwrap().unwrap();
        assert!(!ready.exists());
        assert!(!arm.exists());
        assert!(!armed.exists());
    }

    #[test]
    fn helper_can_release_before_restored_waits_are_armed() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("forkpoint");
        let ready = state.join("ready");
        let restored = state.join("restored");
        let release = state.join("release");
        std::fs::create_dir(&state).unwrap();

        let state_thread = state.clone();
        let ready_thread = ready.clone();
        let restored_thread = restored.clone();
        let release_thread = release.clone();
        let helper = std::thread::spawn(move || {
            run_helper_at(
                ForkpointPaths {
                    state_dir: &state_thread,
                    ready_path: &ready_thread,
                    restored_path: &restored_thread,
                    release_path: &release_thread,
                    arm_path: &state_thread.join("arm"),
                    armed_path: &state_thread.join("armed"),
                },
                Duration::from_secs(60),
                false,
                false,
            )
        });
        wait_for_marker(&ready);
        let generation = marker_generation(&ready);
        assert!(!restored.exists());
        std::fs::write(&release, format!("{RELEASE_PREFIX}{generation}\n")).unwrap();
        helper.join().unwrap().unwrap();
        assert!(!ready.exists());
    }

    #[test]
    fn armable_helper_sleeps_until_capture_and_parks_again_after_disarm() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("forkpoint");
        let ready = state.join("ready");
        let restored = state.join("restored");
        let release = state.join("release");
        let arm = state.join("arm");
        let armed = state.join("armed");
        std::fs::create_dir(&state).unwrap();

        let state_thread = state.clone();
        let ready_thread = ready.clone();
        let restored_thread = restored.clone();
        let release_thread = release.clone();
        let arm_thread = arm.clone();
        let armed_thread = armed.clone();
        let helper = std::thread::spawn(move || {
            run_helper_at(
                ForkpointPaths {
                    state_dir: &state_thread,
                    ready_path: &ready_thread,
                    restored_path: &restored_thread,
                    release_path: &release_thread,
                    arm_path: &arm_thread,
                    armed_path: &armed_thread,
                },
                Duration::from_millis(1),
                false,
                true,
            )
        });
        wait_for_marker(&ready);
        let generation = marker_generation(&ready);
        assert!(!armed.exists());

        std::fs::write(&arm, format!("{ARM_PREFIX}{generation}\n")).unwrap();
        wait_for_marker(&armed);
        assert!(marker_matches(&armed, ARMED_PREFIX, &generation));
        std::fs::remove_file(&arm).unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while armed.exists() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(!armed.exists());
        assert!(!helper.is_finished());

        std::fs::write(&arm, format!("{ARM_PREFIX}{generation}\n")).unwrap();
        wait_for_marker(&armed);
        std::fs::write(&restored, b"restored\n").unwrap();
        std::fs::write(&release, format!("{RELEASE_PREFIX}{generation}\n")).unwrap();
        helper.join().unwrap().unwrap();
        assert!(!ready.exists());
        assert!(!armed.exists());
    }

    #[test]
    fn worker_ready_helper_atomically_publishes_the_lease_token() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("forkpoint");
        let env = temp.path().join("fork-env");
        let marker = state.join("worker-ready");
        let token = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        std::fs::write(
            &env,
            format!("LEARNER=3\n{WORKER_READY_TOKEN_ENV}={token}\n"),
        )
        .unwrap();

        write_worker_ready_at(&state, &env, &marker).unwrap();

        assert_eq!(
            std::fs::read_to_string(marker).unwrap(),
            format!("{token}\n")
        );
        assert!(std::fs::read_dir(state).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with('.')));
    }

    #[test]
    fn worker_ready_helper_rejects_missing_duplicate_or_invalid_tokens() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("forkpoint");
        let env = temp.path().join("fork-env");
        let marker = state.join("worker-ready");
        for contents in [
            "LEARNER=3\n".to_string(),
            format!(
                "{WORKER_READY_TOKEN_ENV}={}\n{WORKER_READY_TOKEN_ENV}={}\n",
                "a".repeat(64),
                "b".repeat(64)
            ),
            format!("{WORKER_READY_TOKEN_ENV}=not-a-token\n"),
        ] {
            std::fs::write(&env, contents).unwrap();
            assert!(write_worker_ready_at(&state, &env, &marker).is_err());
            assert!(!marker.exists());
        }
    }

    #[test]
    fn helper_records_cuda_module_preload_hint() {
        assert_eq!(
            ready_content(true, "0123"),
            "smolvm-forkpoint-v1\ngeneration=0123\ncuda-preload-modules\n"
        );
        assert_eq!(
            ready_content(false, "0123"),
            "smolvm-forkpoint-v1\ngeneration=0123\n"
        );
    }

    #[test]
    fn release_requires_its_generation_or_the_legacy_token() {
        let temp = tempfile::tempdir().unwrap();
        let release = temp.path().join("release");
        std::fs::write(&release, format!("{RELEASE_PREFIX}old\n")).unwrap();
        assert!(!release_matches(&release, "new"));
        assert!(release_matches(&release, "old"));

        std::fs::write(&release, format!("{LEGACY_RELEASE_TOKEN}\n")).unwrap();
        assert!(release_matches(&release, "new"));
    }
}
