# push-guard (v2)

`push-guard.sh` is a Claude Code `PreToolUse` hook for the Bash tool. Agents may push feature branches to `origin` (the `pbrady/shoccs` fork) only. Pushes to `main`, `backup/*`, tags and `upstream` (`lanl/shoccs`) are blocked, and so are deletes and non-lease force pushes.

The sandbox's `excludedCommands` lists only `git push`, `git push *`, `git fetch`, `git fetch *`, `git pull`, `git pull *` and `git ls-remote *`. Those commands run unsandboxed with the owner's ssh key; everything else runs inside the OS sandbox, which has no ssh, cannot read `~/.ssh` and cannot write the project's `.git/config` or `.git/hooks`. A compound command runs unsandboxed only if every subcommand matches an exclusion. The hook makes sure that any command that could run unsandboxed is exactly one safe network git command in an allowed repo. Server-side rulesets remain the real protection for `main`.

v2 replaces v1's segment parser (843 lines) with a single rule: a command that mentions git together with a network verb must be **one plain command**. The v1 files are kept as `push-guard.v1.sh`, `test-push-guard.v1.sh` and `README.v1.md`.

## Files

- `push-guard.sh`: the hook. It is POSIX sh, passes shellcheck, and is tested under bash and dash.
- `test-push-guard.sh`: the test suite. It builds scratch repos with local bare remotes and runs every v1 case plus the v2 cases. It then runs each allowed network command for real and checks that no protected ref moved.

## Install (host, macOS)

1. Copy the hook somewhere agents cannot write, then run the tests.

   ```sh
   mkdir -p ~/.claude/hooks
   cp tools/agents/push-guard.sh ~/.claude/hooks/shoccs-push-guard.sh
   chmod 0755 ~/.claude/hooks/shoccs-push-guard.sh
   ```

2. Register the hook in `~/.claude/settings.json`:

   ```json
   {
     "hooks": {
       "PreToolUse": [
         {
           "matcher": "Bash",
           "hooks": [
             {
               "type": "command",
               "command": "f=\"$HOME/.claude/hooks/shoccs-push-guard.sh\"; if [ ! -f \"$f\" ] || [ ! -r \"$f\" ]; then echo \"push-guard missing or unreadable: $f\" >&2; exit 2; fi; sh \"$f\" --repo /path/to/shoccs --remote origin --protect main --protect 'backup/*' --allow-force-with-lease; rc=$?; case \"$rc\" in 0|2) exit \"$rc\";; *) echo \"push-guard crashed (exit $rc)\" >&2; exit 2;; esac",
               "timeout": 60,
               "onFailure": "block"
             }
           ]
         }
       ]
     },
     "permissions": { "ask": ["Edit(~/.claude/hooks/**)"] }
   }
   ```

   Claude Code runs the command with `sh -c`. The wrapper is fail-closed by itself: a missing or unreadable script exits 2, and any status other than 0 or 2 (127, a signal) becomes 2. Keep `onFailure: "block"` as well, although its effect on `PreToolUse` is unverified. A hook **timeout lets the command run**, so the script bounds its own run time (see the dry-run watchdog and the 100 kB cap below).

3. Run the tests in a scratch directory:

   ```sh
   sh tools/agents/test-push-guard.sh --shell bash --workdir "$TMPDIR/pg"
   sh tools/agents/test-push-guard.sh --shell dash --json python3 --workdir "$TMPDIR/pg"
   sh tools/agents/test-push-guard.sh --sabotage --workdir "$TMPDIR/pg"   # must report N of N
   ```

## Arguments

| flag | meaning |
|---|---|
| `--repo DIR` (repeatable) | Allowed clone. `DIR/.git` must be a directory. Linked worktrees of it are allowed too. |
| `--remote NAME` (repeatable) | Remotes that may receive pushes. The remote must also be configured in the repo. |
| `--protect GLOB` (repeatable) | Protected branch names, matched as shell globs, also against the lowercased name. |
| `--allow-force-with-lease` | Allows `--force-with-lease`, `--force-with-lease=<ref>[:<expect>]` and `--force-if-includes`. |

`--repo`, `--remote` and `--protect` are each required at least once. Any other argument, a flag without a value or a missing required flag is a configuration error. With a configuration error, every network git command is blocked (other commands still pass).

## Algorithm

0. **Input checks.** The input must read cleanly and look like one JSON object (`{...}`); empty, truncated or unreadable input blocks, as does a JSON `\u0000` escape (it would cut the command short in the shell).
1. **Fast path on raw input.** If the raw JSON contains no `git` in any case, no quote or backslash right after a `g` or `i`, and no `\u` escape, the decoded command cannot spell git, so the hook exits 0 without parsing. This takes about 3 ms.
2. **Parse.** The hook reads `tool_input.command` and `cwd` with `jq`, then `python3 -I`, then `plutil` (macOS; the value must be of type string). A parser's output is used only if that parser succeeds. If none of them works, the command is blocked. `PUSH_GUARD_JSON=jq|python3|plutil|none` forces a parser for the tests. A command over 100000 characters blocks, so that the per-character scan stays well inside the hook timeout.
   **ANSI-C escapes.** A command containing `$'` followed anywhere later by a numeric or control escape (`\x..`, `\NNN`, `\u..`, `\U..`, `\c.`) is blocked outright, because bash would decode it into any word (`git $'\x70ush' upstream main`). The scans do not decode escapes; plain `$'\n'` and `$'\t'` stay allowed, and `$'...'`/`$"..."` are otherwise treated like ordinary quotes. The raw-input fast path does not skip a command that contains `$'` or `$"`.
3. **Candidate.** Two awk scans over the lowercased command; either one makes it a candidate. A failed awk, or one that prints neither `y` nor `n`, blocks.
   - **Text scan (A).** Per line (backslash-newline joined), with quotes and backslashes deleted, split on `; & | ( )` and backticks even inside quotes. A segment is a candidate when it holds a token `git` or `…/git` and a verb token, in any order. The verbs are `push`, `fetch`, `pull`, `ls-remote`, `clone`, `daemon`, `--remote`, `remote-*`, `send/fetch/receive/upload-pack`, `http-push/fetch` and `upload-archive`. A dashed `git-<verb>` token also counts. Tokens are runs of `[a-z0-9._/-]`, so `alias.zz=push` and `$IFS` splits are seen. A commit heredoc whose later lines say "push" passes, because git is on another line.
   - **Word scan (B).** The whole command split into shell words at **unquoted** blanks and `< >`, and into simple commands at unquoted `; & | ( )`, backticks and newlines; quotes are removed. A simple command is a candidate when one word is `git` (or `…/git`, `=git`) and one word is a verb (also `NAME=verb`, `--verb=…`). This catches a quoted separator between git and its verb, such as `git -c "x=;" push origin main` or a quoted newline, which scan A splits apart.

   Anything that is not a candidate exits 0. `git status` and `git log | grep push` take about 8 ms; a non-git command about 4 ms.
4. **One plain command.** The whole command must use only `A-Za-z0-9 ._/:@=+,%^~-`, which rules out quotes, `\`, `$`, backticks, `; & | < > ( ) { } * ? [ ] ! #`, tabs, newlines, CR and non-ASCII. Runs of spaces are fine. The first word must be exactly `git`, and the second word must be `push`, `fetch`, `pull` or `ls-remote`. There is no `-C`, `-c`, `--no-pager` or any other global option, no env prefix, no wrapper, no path to git, and no `GIT`. Otherwise the command is blocked, with a message that tells the agent to run one plain command from the repo directory.
5. **Session environment.** The command is blocked if any of these is set in the hook's environment: `GIT_DIR`, `GIT_WORK_TREE`, `GIT_COMMON_DIR`, `GIT_INDEX_FILE`, `GIT_OBJECT_DIRECTORY`, `GIT_ALTERNATE_OBJECT_DIRECTORIES`, `GIT_NAMESPACE`, `GIT_CONFIG`, `GIT_CONFIG_PARAMETERS`, `GIT_CONFIG_COUNT` or `GIT_EXEC_PATH`. A command line cannot set them, because rule 4 forbids env prefixes. A set value would have to come from the session environment that the hook and the command share, and it would make the repo checks below meaningless.
6. **Repository.** The hook runs `git -C <cwd> rev-parse --path-format=absolute --git-dir --git-common-dir --show-toplevel` and resolves each path physically. The common dir must equal `realpath(<repo>/.git)` for an allowed repo. Then one of these must hold:
   - **Main clone:** the git dir equals the common dir, and the top level equals the repo.
   - **Linked worktree:** the git dir sits directly in `<repo>/.git/worktrees/`, and its `gitdir` back-pointer names `<toplevel>/.git`.

   Before either check, the hook also requires that the top level is **outside every git dir**: no path component of the top level may be `.git`, the top level and cwd must not be inside the git dir or the common dir, the top level must not be inside any allowed repo's `.git`, and it must not be a strict ancestor of any allowed repo. A consequence is that one allowed repo cannot be nested inside another allowed repo's work tree.

   These checks catch a rewritten `commondir`, a `.git` file that borrows the allowed git dir (`redir`, `redir2`), nested repos, `.git/` itself, and a hand-written linked worktree rooted in a git dir (E6).
7. **Per-verb rules.** Options must match an exact list of tokens. Abbreviations, `=` variants and bundled short flags (`-uv`) are refused.
   - **push:** The options allowed are `-u --set-upstream -v --verbose -q --quiet -n --dry-run --porcelain --progress --atomic`, plus the lease options when they are enabled. The first positional is the remote. It must be on the `--remote` list and configured (`remote.<r>.url`). There must be one or more refspecs, each exactly `B:refs/heads/B`. `B` must be a plain branch name (`[A-Za-z0-9._/-]`, no leading `-` or `.`, no `..`, `//` or `.lock`, and not `HEAD`, `refs/…`, `heads/…`, `tags/…` or `remotes/…`). It must not be protected, and it must exist as `refs/heads/B`. The hook then runs a **dry run of the agent's literal arguments** plus `--dry-run --porcelain --no-verify`, using `ssh -o BatchMode=yes` unless `GIT_SSH_COMMAND`, `GIT_SSH` or `core.sshCommand` is set. The dry run must satisfy all of these:
     - every `To` URL is a fetch URL of the remote, which catches a `pushurl` that points elsewhere;
     - every destination is one of the requested `refs/heads/B` and is not protected;
     - no line is a `-` (delete);
     - `+` (forced) lines appear only with `--force-with-lease` (`--force-if-includes` alone does not count);
     - there is at least one ref line and one `To` line;
     - a failed dry run passes only if git finished (`Done`) and rejected a ref (`!`, e.g. non-fast-forward); anything else, including a killed dry run, blocks.

     The dry run runs under a 40 s watchdog (`PUSH_GUARD_DRY_RUN_SECS` overrides it for the tests), and the default ssh command adds `ConnectTimeout=15`, so a hung network blocks instead of running the hook into its 60 s timeout.
   - **fetch:** `--no-recurse-submodules` (or `--recurse-submodules=no`) is **required**. The other options allowed are `-v --verbose -q --quiet --prune -p --tags --no-tags --all --dry-run`. The positionals are an optional configured remote name followed by plain branch names, with no `:` or `+`. `--all` takes no positionals. Without a remote, the remote git would pick (`branch.<cur>.remote`, else `origin`) must be a configured remote name.
   - **pull:** `--ff-only` and `--no-recurse-submodules` are **required**. The other options allowed are `-v --verbose -q --quiet --prune -p --tags --no-tags`. The positionals are an optional remote and at most one plain branch. `--rebase`, `--all`, `-s`, `-X` and `--dry-run` are refused. The default remote is checked as for fetch.
   - **ls-remote:** The options allowed are `-h --heads -t --tags --refs -q --quiet --symref`. A configured remote is required, and optional plain patterns may follow it.

## Decisions (spec "decide" items and conflicts)

| item | decision | why |
|---|---|---|
| Refspec form | `B:refs/heads/B` only. A bare `B` is **blocked**, which departs from the spec's "plain branch name, no `:`" | The spec also says to keep all v1 block cases, and v1 blocks `git push origin feat`. Two experiments show bare names are unsafe. (a) If `refs/heads/sym` is a symref to `main`, `git push origin sym` pushes `refs/heads/sym:refs/heads/main`. A sandboxed background job can create that symref after the hook's dry run, so a hook-time check cannot close the race. (b) With `remote.origin.push = refs/heads/feat2:refs/heads/main`, `git push origin feat2` pushes to `main`. An explicit same-name destination does neither: the worst a race can do is change which commit lands on `B`. |
| `--no-verify` | blocked | v1 blocked it. The pre-push hook is the owner's policy (sandbox-protected `.git/hooks`), and the agent should not skip it. The hook's own dry run always passes `--no-verify`, because without it the dry run executes `pre-push`, which was confirmed. |
| Candidate rule | text scan per segment (`git` token plus verb token in any order, after deleting quotes and backslashes) OR quote-aware word scan per simple command | This is tighter than "git and a verb anywhere": heredoc commit bodies that mention "push" on later lines pass. It is looser than "verb right after git", which would miss `FOO=push git zz` and `git -c alias.zz=push zz`. Backslash-newline is joined first, or `git \⏎push` would slip through. Known over-blocks: `git commit -m "fix pull request"` and `git log --grep push`. Commit with `-F file` or a heredoc instead. |
| `clone`, `archive --remote`, `daemon`, `remote-*`, `*-pack` | blocked | They are not excluded, so they run sandboxed anyway. Blocking them is one regex alternative and keeps the v1 block cases. Clone repos yourself. |
| Aliases and external subcommands (`git p origin main`, `git sh`, `env git zz …`) | **allowed (exit 0)**, a change from v1 | Spec step 7: only network verbs are candidates. `git <alias>` never matches a `git push *`-style exclusion, so it runs sandboxed, with no ssh and no credentials. Git ignores aliases that shadow `push`, `fetch`, `pull` and `ls-remote`. The 9 v1 cases are kept in the tests as `allow` with a comment. |
| Submodules | `--no-recurse-submodules` is required on **every** fetch and pull, not only when `.gitmodules` exists | `.gitmodules` and gitlinks can arrive in the fetched commits. With on-demand recursion, git then fetches inside a plain nested clone, whose own `.git/config` is a sandbox-writable work-tree file. v1 round 5 confirmed that the attacker's `uploadpack` ran unsandboxed. A local `.gitmodules` check is incomplete. shoccs has no `.gitmodules`, so the only cost is the flag. |
| `pull` | allowed only as `git pull --ff-only --no-recurse-submodules [remote [branch]]` | Merge and rebase can run hooks (`post-merge`) and filters, but hooks live in protected `.git/hooks`, and filter and merge drivers must be defined in config. A `.gitattributes` naming `filter=evil merge=evil diff=evil` with no config ran nothing on a pull (experiment E5). `--ff-only` rules out merge commits and strategies. Residual risk: a fast-forward writes work-tree files unsandboxed, including sandbox-protected paths such as `.claude/` and `.github/workflows/`, from a branch the agent pushed. Consider dropping `git pull *` from `excludedCommands` and using fetch followed by a sandboxed `git merge --ff-only`. |
| `fetch --all`, `--tags`, `--dry-run` | allowed | These are read-only and fetch configured remotes only. |
| fetch refspecs with `:` or `+` | blocked | `origin main:main` writes local branches, so the hook keeps fetch to plain names. |
| `ls-remote` without a remote | blocked | Making the remote explicit keeps the rule simple. |
| Remote for fetch, pull and ls-remote | any configured remote name | This keeps reading from `upstream` possible. URLs and paths are never accepted. |
| Env vars (`GIT_DIR` …) | block if set in the hook env | See algorithm step 5. `GIT_SSH`, `GIT_SSH_COMMAND` and `GIT_CONFIG_GLOBAL` are not blocked; they are the owner's session settings. |
| `--allow-project-dir` | removed | It is not in the v2 argument list. Passing it is now a configuration error, which blocks. |
| Fast path before parsing | on the raw JSON | This is equivalent to the spec's "command has no git" check. It needs no parser, and it is sound because only `\u` escapes, quotes and backslashes can hide letters. |

## Experiments (git 2.50.1, scratch repos with local bare remotes)

- **E1:** a repo with `core.fsmonitor` and `pre-push`, `reference-transaction`, `post-checkout`, `post-index-change`, `pre-auto-gc` and `post-merge` hooks set to a payload. `rev-parse --git-dir/--git-common-dir/--show-toplevel`, `config --get`, `show-ref --verify`, `symbolic-ref`, `remote get-url`, `ls-remote` and `push --dry-run --porcelain --no-verify` ran **no payload**. `push --dry-run` without `--no-verify` ran `pre-push`. `git status` (the control) ran `fsmonitor` and `post-index-change`.
- **E2:** symref race. `git push origin sym` with `sym` pointing to `main` gave `refs/heads/sym:refs/heads/main`; `sym:refs/heads/sym` gave `refs/heads/sym:refs/heads/sym`.
- **E3:** with `remote.origin.push=refs/heads/feat2:refs/heads/main`, `git push origin feat2` gave `refs/heads/feat2:refs/heads/main`, while the explicit form went to `feat2`. A `remote.origin.mirror=true` makes a push with refspecs fail, so the dry run blocks it.
- **E4:** `plutil -extract tool_input.command raw -o - -` reads JSON on stdin. It extracts numbers as text, so the hook checks `plutil -type` = `string` first.
- **E5:** see the `pull` row above.
- **E6 (FRT2, gitdir worktree):** all sandboxed writes outside `.git/config` and `.git/hooks`: commits `wbase` (empty tree) and `evil` (adds `hooks/post-merge`, mode 755), `evil` pushed with the allowed form; then `.git/worktrees/w/{HEAD,commondir,gitdir}` (`ref: refs/heads/wbase`, `../..`, `<repo>/.git/.git`) and the gitfile `<repo>/.git/.git` pointing at `.git/worktrees/w`. From cwd `<repo>/.git`, `git rev-parse` reports git dir `<repo>/.git/worktrees/w`, common dir `<repo>/.git` and top level `<repo>/.git`; `--is-inside-git-dir` prints `false`, so that check would not help. The v2 hook allowed `git pull --ff-only --no-recurse-submodules origin evil`, and the unsandboxed fast-forward wrote `<repo>/.git/hooks/post-merge` and ran it at once. The same works for any top level inside a git dir, and a top level that is an ancestor of the repo would let a pull write into the repo's work tree from outside. Fixed by the "outside every git dir" rule in step 6; the tests build all four layouts (`frt2` clone, `outer/inner`) and also run the attack live if the hook ever allows it.

## Known limitations

- The hook is textual, not a shell parser, and it leans toward blocking. See the over-blocks in the candidate row above. The word scan does not model substitutions nested inside double quotes (`echo "$(git -c 'x=;' push …)"`); like other hidden forms, such a command does not match the literal `excludedCommands` entries and runs sandboxed.
- A hook timeout fails open. The script's own bounds (40 s dry-run watchdog, 100 kB cap) keep it well inside the 60 s timeout; a non-push command never touches the network.
- Obfuscation that hides the literal word git or the verb (`$(printf gi)t push`, `{g,}it`, scripts, `make`, `xargs`, `sh -c "$v"`) is not detected. Such commands do not match the literal `excludedCommands` entries, so they run sandboxed. This holds only while `excludedCommands` lists only literal `git push|fetch|pull|ls-remote` forms, never `sh *`, `bash *`, `env *` or `make *`.
- The hook trusts repo config (`remote.*.url`, `core.sshCommand`, `insteadOf`, `pull.rebase`) and relies on the sandbox to deny writes to `.git/config`, `.git/hooks` and the user's git config. The dry run still catches a `pushurl` that points elsewhere. `extensions.worktreeConfig` would make `.git/worktrees/*/config.worktree` (sandbox-writable) part of the config, but enabling it needs `.git/config`.
- **Worktree placement.** A linked worktree may still sit anywhere outside a git dir, including inside the allowed work tree (e.g. `.claude/worktrees/*`, which is the intended use) or under a sandbox-protected work-tree directory such as `.claude/`. A pull into such a worktree writes files there unsandboxed; this is the same residual risk as the `pull` row above (dropping `git pull *` from `excludedCommands` removes it). Git itself refuses tree paths that contain a `.git` component and does not write through a symlinked leading directory (standard git behaviour, not retested here).
- **Races.** Local refs are sandbox-writable. With `B:refs/heads/B`, a race can change only which commit goes to `B`, never the destination.
- Tags cannot be pushed by agents at all.
