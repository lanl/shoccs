#!/bin/sh
# push-guard.sh (v2): Claude Code PreToolUse hook (matcher "Bash").
#
#   sh push-guard.sh --repo DIR [--repo ...] --remote NAME [--remote ...] \
#      --protect GLOB [--protect ...] [--allow-force-with-lease]
#
# Reads the hook JSON on stdin. Exit 0 = allow (normal permissions and the
# sandbox still apply), exit 2 = block (reason on stderr). Every error path
# blocks: bad input, no JSON parser, git missing, an unexpected status.
#
# sandbox.excludedCommands lists only `git push|fetch|pull|ls-remote`; those
# run unsandboxed with the owner's ssh key. Any command that mentions git
# together with a network verb must therefore be exactly ONE plain command:
#   git push [opts] <allowed remote> <branch>:refs/heads/<branch>...
#   git fetch --no-recurse-submodules [opts] [<remote> [<branch>...]]
#   git pull --ff-only --no-recurse-submodules [opts] [<remote> [<branch>]]
#   git ls-remote [opts] <remote> [<pattern>...]
# run from an allowed clone or a linked worktree of it. No chains, cd, git -C,
# git -c, global options, quotes, $, redirects or any shell syntax.
# POSIX sh; tested under bash 3.2 and dash, BSD and GNU userland.

# shellcheck disable=SC2154
# SC2154: rc is assigned inside the EXIT trap string.
set -u
set -f
LC_ALL=C
export LC_ALL
unset CDPATH
# Fail closed: an unexpected status (shell error, unbound variable) becomes 2.
trap 'rc=$?; if [ "$rc" != 0 ] && [ "$rc" != 2 ]; then echo "push-guard: BLOCKED: internal error (status $rc)" >&2; exit 2; fi' EXIT

nl='
'
tab=$(printf '\t')
REPOS='' REMOTES='' PROTECT='' ALLOW_LEASE=0 CONFIG_ERR=''
while [ $# -gt 0 ]; do
  case $1 in
    --repo|--remote|--protect)
      if [ $# -lt 2 ] || [ -z "$2" ]; then CONFIG_ERR="$1 needs a value"; shift; continue; fi
      case $1 in
        --repo) REPOS=$REPOS$2$nl ;;
        --remote) REMOTES=$REMOTES$2$nl ;;
        *) PROTECT=$PROTECT$2$nl ;;
      esac
      shift 2 ;;
    --allow-force-with-lease) ALLOW_LEASE=1; shift ;;
    *) CONFIG_ERR="unknown argument: $1"; shift ;;
  esac
done
# A missing flag would silently widen the policy (no --protect = no
# protected branches), so all three are required.
[ -n "$REPOS" ] || CONFIG_ERR=${CONFIG_ERR:-"no --repo given"}
[ -n "$REMOTES" ] || CONFIG_ERR=${CONFIG_ERR:-"no --remote given"}
[ -n "$PROTECT" ] || CONFIG_ERR=${CONFIG_ERR:-"no --protect given"}

block() {
  printf 'push-guard: BLOCKED: %s\n' "$1" >&2
  printf 'push-guard: run ONE plain command from the repo directory (no &&, ;, |, cd, -C, quotes, $): git push [-u] <remote> <branch>:refs/heads/<branch> | git fetch --no-recurse-submodules [<remote>] | git pull --ff-only --no-recurse-submodules [<remote> [<branch>]] | git ls-remote <remote>\n' >&2
  exit 2
}
in_list() { # word newline-list
  case $nl$2 in *"$nl$1$nl"*) return 0 ;; esac
  return 1
}
is_protected() { # branch name, also matched lowercased
  _lc=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  _ifs=$IFS; IFS=$nl
  for _p in $PROTECT; do
    # shellcheck disable=SC2254
    case $1 in $_p) IFS=$_ifs; return 0 ;; esac
    # shellcheck disable=SC2254
    case $_lc in $_p) IFS=$_ifs; return 0 ;; esac
  done
  IFS=$_ifs
  return 1
}
rp() { (cd -P -- "$1" 2>/dev/null && pwd -P); }
gitc() { git -C "$cwd" "$@" 2>/dev/null </dev/null; }
is_remote() { # configured remote name (never a URL or path)
  case $1 in ''|-*|*[!A-Za-z0-9._-]*) return 1 ;; esac
  [ -n "$(gitc config --get "remote.$1.url")" ]
}
plain_name() { # plain branch name: no ':' '+' leading '-', not HEAD or refs/...
  case $1 in
    ''|-*|.*|*/|*..*|*//*|*.lock|HEAD|refs/*|heads/*|tags/*|remotes/*|*[!A-Za-z0-9._/-]*) return 1 ;;
  esac
  return 0
}
anon_url() { # strip user@ the way git prints "To <url>"
  case $1 in
    *://*@*) _r=${1#*://}; printf '%s://%s' "${1%%://*}" "${_r#*@}" ;;
    *://*) printf '%s' "$1" ;;
    *@*:*) case ${1%%:*} in *@*) printf '%s' "${1#*@}" ;; *) printf '%s' "$1" ;; esac ;;
    *) printf '%s' "$1" ;;
  esac
}

# ------------------------------------------------------------- 1. input ---
input=$(cat) || block "could not read the hook input"
# Claude Code always sends one JSON object; anything else (empty, truncated)
# could hide a command, so it blocks.
case $input in
  '{'*'}') ;;
  *) block "hook input is not a JSON object" ;;
esac
# A JSON-escaped NUL would cut the command short in $(...) below.
case $input in *'\u0000'*) block "hook input contains a NUL escape" ;; esac
# 2. Fast path on the raw JSON: no "git" (any case), no quote or backslash
# after g/i (g''it, g\it, JSON-escaped g\"\"it) and no \u escape means no
# quoting trick in the decoded command spells git. ($-expansion tricks such
# as $(printf gi)t are never detected; they do not match excludedCommands
# and so stay sandboxed. See README "Known limitations".)
case $input in
  *[Gg][Ii][Tt]*|*[GgIi]\'*|*[GgIi]\\*|*'\u'*|*\$\'*|*\$\\\"*) ;;
  *) exit 0 ;;
esac

get_field() { # command | cwd ; jq, then python3 -I, then plutil (macOS)
  _key=$1; [ "$1" = command ] && _key=tool_input.command
  case ${PUSH_GUARD_JSON:-auto} in
    jq|python3|plutil) _ps=${PUSH_GUARD_JSON} ;;
    none) return 1 ;;
    *) _ps='jq python3 plutil' ;;
  esac
  for _p in $_ps; do
    command -v "$_p" >/dev/null 2>&1 || continue
    # Each parser's output is kept only if that parser succeeds.
    case $_p in
      jq) _v=$(printf '%s' "$input" | jq -je ".$_key | strings" 2>/dev/null) ;;
      python3) _v=$(printf '%s' "$input" | python3 -I -c '
import json, sys
d = json.load(sys.stdin)
v = d["tool_input"]["command"] if sys.argv[1] == "command" else d["cwd"]
sys.exit(1) if not isinstance(v, str) else sys.stdout.write(v)' "$1" 2>/dev/null) ;;
      plutil)
        [ "$(printf '%s' "$input" | plutil -type "$_key" - 2>/dev/null)" = string ] &&
          _v=$(printf '%s' "$input" | plutil -extract "$_key" raw -o - - 2>/dev/null) ;;
    esac && { printf '%s' "$_v"; return 0; }
  done
  return 1
}
cmd=$(get_field command) || block "could not parse tool_input.command (need jq, python3 or plutil)"
# The scan below is per character; a huge command must not run the hook into
# its timeout (a timed-out hook lets the command run). 100 kB takes ~0.5 s.
[ "${#cmd}" -le 100000 ] || block "command longer than 100000 characters; put long text in a file"
# bash ANSI-C quoting can spell any word with escapes ($'\x70ush',
# $'\160ush', $'\u0070'), which neither scan below decodes. Rather than
# decode, block a command that has $' and, anywhere after it, a backslash
# escape that encodes a character by number or control code. Plain $'\n',
# $'\t' stay allowed.
case $cmd in
  *"\$'"*)
    if printf '%s' "$cmd" | tr '\n' ' ' | grep -Eq "\\\$'.*\\\\[0-7xuUcE]"; then
      block "ANSI-C escapes that encode characters (\$'\\x..', \$'\\NNN', \$'\\u..') are not allowed; write the text literally"
    fi ;;
esac

# ---------------------------------------- 3. git-network candidate? ---
# Two awk scans over the lowercased command; either one makes a candidate.
#  A (text):  per line (backslash-newline joined), quotes and backslashes
#     deleted, split at ; & | ( ) and backticks EVEN inside quotes. A segment
#     is a candidate if it has a token "git" or ".../git" and a network-verb
#     token, in any order, or a dashed git-<verb>. Tokens are runs of
#     [a-z0-9._/-], so alias.zz=push and FOO=push count. Commit-message
#     heredoc lines that say "push" pass, because git is on another line.
#  B (words): the whole command split into shell words and simple commands
#     at UNQUOTED blanks, < > ; & | ( ) ` and newlines, quotes removed. A
#     simple command is a candidate if one word is git (or .../git) and one
#     word is a verb (also as NAME=verb or --verb=...). This catches a quoted
#     separator between git and its verb (git -c "x=;" push), which A splits.
# Not a shell parser: $-expansions and substitutions nested inside double
# quotes are not modelled (see README "Known limitations"). Prints y or n;
# a failed or garbled scan blocks.
cand=$(printf '%s\n' "$cmd" | awk -v sq="'" -v dq='"' '
  { lines[++nlines] = tolower($0) }
  END {
    hit = 0; s = ""; buf = ""
    for (k = 1; k <= nlines; k++) {
      s = s lines[k] "\n"
      line = lines[k]
      if (sub(/\\$/, "", line)) { buf = buf line; continue }
      if (scan_text(buf line)) hit = 1
      buf = ""
    }
    if (buf != "" && scan_text(buf)) hit = 1
    if (scan_words(s)) hit = 1
    print (hit ? "y" : "n")
  }
  function verb(t) {
    return t ~ /^(push|fetch|pull|ls-remote|clone|daemon|--remote|remote-.*|(send|fetch|receive|upload)-pack|http-(push|fetch)|upload-archive)$/
  }
  function dashed(t) { # git-push, /usr/libexec/git-core/git-fetch-pack ...
    if (t !~ /(^|\/)git-/) return 0
    sub(/^.*git-/, "", t); return verb(t)
  }
  function scan_text(s,   nseg, segs, i, n, toks, j, g, v) {
    gsub(/["\\]/, "", s); gsub(sq, "", s)
    nseg = split(s, segs, /[;&|()`]/)
    for (i = 1; i <= nseg; i++) {
      n = split(segs[i], toks, /[^a-z0-9._\/-]+/); g = 0; v = 0
      for (j = 1; j <= n; j++) {
        if (toks[j] == "git" || toks[j] ~ /\/git$/) g = 1
        else if (verb(toks[j])) v = 1
        else if (dashed(toks[j])) return 1
      }
      if (g && v) return 1
    }
    return 0
  }
  function word(w,   t) { # 1 = git, 2 = verb, 3 = dashed git-verb, 0 = other
    if (w ~ /(^|[\/=])git$/) return 1
    if (dashed(w)) return 3
    if (verb(w)) return 2
    t = w; sub(/^.*=/, "", t); if (verb(t)) return 2
    t = w; sub(/=.*$/, "", t); if (verb(t)) return 2
    return 0
  }
  function scan_words(s,   n, i, c, q, w, k, g, v) {
    n = length(s); q = ""; w = ""; g = 0; v = 0
    for (i = 1; i <= n; i++) {
      c = substr(s, i, 1)
      if (c == "\\" && q != sq) { i++; c = substr(s, i, 1); if (c != "\n") w = w c; continue }
      if (c == "\\") continue
      # $'...' and $"..." quote like '...' and "..." (escape-encoded $'...'
      # content is blocked before the scan)
      if (c == "$" && q == "" && (substr(s, i + 1, 1) == sq || substr(s, i + 1, 1) == dq)) continue
      if (c == sq && q != dq) { q = (q == "") ? sq : ""; continue }
      if (c == dq && q != sq) { q = (q == "") ? dq : ""; continue }
      if (q != "" || !index(" \t<>;&|()`\n", c)) { w = w c; continue }
      k = word(w); w = ""
      if (k == 3) return 1
      if (k == 1) g = 1
      if (k == 2) v = 1
      if (g && v) return 1
      if (index(";&|()`\n", c)) { g = 0; v = 0 }
    }
    k = word(w)
    return k == 3 || (g || k == 1) && (v || k == 2)
  }') || block "candidate scan (awk) failed"
case $cand in
  n) exit 0 ;;
  y) ;;
  *) block "candidate scan (awk) gave no answer" ;;
esac
[ -z "$CONFIG_ERR" ] || block "hook configuration error: $CONFIG_ERR"
command -v git >/dev/null 2>&1 || block "git is not on the hook's PATH"

# ------------------------------------------- 4. exactly one plain command ---
case $cmd in
  *[!A-Za-z0-9\ ._/:@=+,%^~-]*)
    block "network git commands must be plain words: no quotes, \$, backticks, ; & | < > ( ) { } * ? [ ] ! # \\, tabs, newlines or non-ASCII" ;;
esac
# shellcheck disable=SC2086
set -- $cmd
[ "${1:-}" = git ] || block "the command must start with the word git (no env assignments, wrappers or paths)"
verb=${2:-}
case $verb in
  push|fetch|pull|ls-remote) shift 2 ;;
  -*) block "no git global options (-C, -c, --git-dir, ...); cd is not allowed either: start the session in the repo" ;;
  *) block "only git push/fetch/pull/ls-remote may mention a network verb; 'git $verb' with push/fetch/pull/clone words must be run by the owner or reworded (e.g. commit with -F)" ;;
esac

# Variables that redirect git away from the cwd's repository or its config.
for _v in GIT_DIR GIT_WORK_TREE GIT_COMMON_DIR GIT_INDEX_FILE GIT_OBJECT_DIRECTORY \
          GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_NAMESPACE GIT_CONFIG GIT_CONFIG_PARAMETERS \
          GIT_CONFIG_COUNT GIT_EXEC_PATH; do
  eval "_x=\${$_v-}"
  [ -z "$_x" ] || block "$_v is set in the session environment"
done

# ------------------------------------------------------ 5. repository ---
cwd=$(get_field cwd) || cwd=''
case $cwd in /*) ;; *) block "hook input has no absolute cwd" ;; esac
cwd=$(rp "$cwd")
[ -n "$cwd" ] || block "cwd does not exist"
info=$(gitc rev-parse --path-format=absolute --git-dir --git-common-dir --show-toplevel) ||
  block "cwd is not inside a git work tree"
case $info in *"$nl"*"$nl"*) ;; *) block "could not resolve the repository of cwd" ;; esac
gd=$(rp "${info%%"$nl"*}"); _r=${info#*"$nl"}
cd_=$(rp "${_r%%"$nl"*}"); top=$(rp "${_r#*"$nl"}")
[ -n "$gd" ] && [ -n "$cd_" ] && [ -n "$top" ] || block "could not resolve the repository of cwd"
# The work tree (and cwd) must lie outside every git dir. A linked worktree
# whose top level is <repo>/.git itself (gitfile <repo>/.git/.git) would let
# an unsandboxed pull check out attacker files into .git/hooks.
case "/$top/" in */.git/*) block "the work tree lies inside a .git directory" ;; esac
for _p in "$gd" "$cd_"; do
  case "$top/" in "$_p/"*) block "the work tree lies inside the git dir" ;; esac
  case "$cwd/" in "$_p/"*) block "cwd lies inside the git dir" ;; esac
done
ok=0
_ifs=$IFS; IFS=$nl
for _a in $REPOS; do
  IFS=$_ifs
  _ar=$(rp "$_a")
  # no work tree may contain an allowed repo (or sit inside its .git)
  if [ -n "$_ar" ]; then
    case "$_ar/" in "$top/"?*) block "the work tree contains the allowed repo $_ar" ;; esac
    case "$top/" in "$_ar/.git/"*) block "the work tree lies inside $_ar/.git" ;; esac
  fi
  if [ -n "$_ar" ] && [ -d "$_ar/.git" ] && [ "$cd_" = "$(rp "$_ar/.git")" ] && [ -n "$cd_" ]; then
    if [ "$gd" = "$cd_" ] && [ "$top" = "$_ar" ]; then ok=1
    elif [ "$(rp "${gd%/*}")" = "$cd_/worktrees" ] && [ -f "$gd/gitdir" ]; then
      # worktree: its gitdir back-pointer must name this work tree's .git
      _back=$(cat "$gd/gitdir" 2>/dev/null)
      [ "${_back##*/}" = .git ] && [ "$(rp "${_back%/*}")" = "$top" ] && ok=1
    fi
  fi
  IFS=$nl
done
IFS=$_ifs
[ "$ok" = 1 ] || block "cwd is not an allowed clone or a linked worktree of one"

# ------------------------------------------------ 6. per-verb checks ---
need_default_remote() { # remote git uses when none is named
  _b=$(gitc symbolic-ref -q --short HEAD) || _b=''
  _dr=''
  [ -n "$_b" ] && _dr=$(gitc config --get "branch.$_b.remote")
  [ -n "$_dr" ] || _dr=origin
  is_remote "$_dr" || block "the default remote '$_dr' is not a configured remote name; name the remote explicitly"
}

case $verb in
fetch|pull)
  _nors=0 _ff=0 _all=0 _pos=0
  for _t do
    case $_t in
      --no-recurse-submodules|--recurse-submodules=no) _nors=1 ;;
      --ff-only) [ "$verb" = pull ] || block "fetch: option $_t is not allowed"; _ff=1 ;;
      --all) [ "$verb" = fetch ] || block "pull --all is not allowed"; _all=1 ;;
      -v|--verbose|-q|--quiet|--prune|-p|--tags|--no-tags) ;;
      --dry-run) [ "$verb" = fetch ] || block "pull: option $_t is not allowed" ;;
      -*) block "$verb: option $_t is not allowed (exact allowlist; abbreviations are refused)" ;;
      *)
        _pos=$((_pos + 1))
        if [ "$_pos" = 1 ]; then
          is_remote "$_t" || block "'$_t' is not a configured remote name (URLs and paths are refused)"
        else
          plain_name "$_t" || block "'$_t' must be a plain branch name (no ':', '+', refs/, HEAD)"
          [ "$verb" = pull ] && [ "$_pos" -gt 2 ] && block "pull takes at most one branch"
        fi ;;
    esac
  done
  [ "$_nors" = 1 ] || block "git $verb must include --no-recurse-submodules (submodule fetches run with work-tree-controlled config)"
  [ "$verb" = fetch ] || [ "$_ff" = 1 ] || block "git pull must include --ff-only (or fetch, then merge/rebase as a separate sandboxed command)"
  [ "$_all" = 1 ] && [ "$_pos" -gt 0 ] && block "fetch --all takes no remote or refspec"
  [ "$_pos" = 0 ] && [ "$_all" = 0 ] && need_default_remote
  exit 0 ;;
ls-remote)
  _pos=0
  for _t do
    case $_t in
      -h|--heads|-t|--tags|--refs|-q|--quiet|--symref) ;;
      -*) block "ls-remote: option $_t is not allowed" ;;
      *)
        _pos=$((_pos + 1))
        if [ "$_pos" = 1 ]; then
          is_remote "$_t" || block "'$_t' is not a configured remote name (URLs and paths are refused)"
        else
          plain_name "$_t" || block "ls-remote pattern '$_t' must be a plain name"
        fi ;;
    esac
  done
  [ "$_pos" -ge 1 ] || block "git ls-remote must name a configured remote"
  exit 0 ;;
esac

# push
_lease=0 _remote='' _dsts=''
for _t do
  case $_t in
    -u|--set-upstream|-v|--verbose|-q|--quiet|-n|--dry-run|--porcelain|--progress|--atomic) ;;
    --force-if-includes)
      [ "$ALLOW_LEASE" = 1 ] || block "$_t is not allowed" ;;
    --force-with-lease|--force-with-lease=*)
      [ "$ALLOW_LEASE" = 1 ] || block "$_t is not allowed"
      case $_t in
        --force-with-lease=*)
          _lv=${_t#--force-with-lease=}
          plain_name "${_lv%%:*}" || block "bad --force-with-lease ref"
          case ${_lv#"${_lv%%:*}"} in ''|:[A-Za-z0-9._/-]*) ;; *) block "bad --force-with-lease value" ;; esac
          case ${_lv#*:} in *:*) block "bad --force-with-lease value" ;; esac ;;
      esac
      _lease=1 ;;
    -*) block "push: option $_t is not allowed (no -f/--force/--all/--mirror/--tags/--delete/--prune/-o/--no-verify; exact allowlist)" ;;
    *)
      if [ -z "$_remote" ]; then
        in_list "$_t" "$REMOTES" || block "remote '$_t' is not an allowed push remote"
        is_remote "$_t" || block "remote '$_t' is not configured in this repo"
        _remote=$_t
      else
        _b=${_t%%:*}
        [ "$_t" = "$_b:refs/heads/$_b" ] ||
          block "refspec '$_t' must be written <branch>:refs/heads/<branch> with the same branch name (implicit destinations follow symrefs and remote.<name>.push)"
        plain_name "$_b" || block "'$_b' is not a plain branch name"
        is_protected "$_b" && block "'$_b' is a protected branch"
        gitc show-ref --verify --quiet "refs/heads/$_b" || block "'$_b' is not an existing local branch"
        _dsts=$_dsts"refs/heads/$_b$nl"
      fi ;;
  esac
done
[ -n "$_dsts" ] || block "git push needs a remote and at least one <branch>:refs/heads/<branch> refspec"

# Dry run of the agent's literal arguments (--no-verify: never runs the
# pre-push hook). Never prompt: batch-mode ssh with a connect timeout unless
# the owner configured an ssh command. A hook that times out lets the command
# run, so a watchdog kills the dry run after 40 s (hook timeout: 60 s); a
# killed or unfinished dry run blocks below.
if [ -z "${GIT_SSH_COMMAND:-}" ] && [ -z "${GIT_SSH:-}" ] && [ -z "$(gitc config --get core.sshCommand)" ]; then
  GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15'; export GIT_SSH_COMMAND
fi
GIT_TERMINAL_PROMPT=0; export GIT_TERMINAL_PROMPT
dry_run() { # prints the porcelain output, then a last line "rc=<status>"
  git -C "$cwd" push --dry-run --porcelain --no-verify "$@" 2>/dev/null </dev/null &
  _g=$!
  ( sleep "${PUSH_GUARD_DRY_RUN_SECS:-40}"; kill "$_g" ) >/dev/null 2>&1 </dev/null &
  _w=$!
  wait "$_g"; _st=$?
  kill "$_w" 2>/dev/null
  printf '\nrc=%s\n' "$_st"
}
_out=$(dry_run "$@")
_rc=${_out##*rc=}; _out=${_out%rc=*}
case $_rc in ''|*[!0-9]*) block "could not read the dry run's exit status" ;; esac
_urls=''
for _u in $(gitc remote get-url --all "$_remote"); do _urls=$_urls$_u$nl$(anon_url "$_u")$nl; done
_refs=0 _tos=0 _rejected=0 _done=0
while IFS= read -r _line; do
  case $_line in
    Done) _done=1 ;;
    'To '*) _tos=$((_tos + 1))
      in_list "${_line#To }" "$_urls" || block "dry run pushes to ${_line#To }, not the fetch URL of '$_remote' (pushurl?)" ;;
    *"$tab"*)
      _refs=$((_refs + 1))
      _flag=${_line%%"$tab"*}; _pair=${_line#*"$tab"}; _pair=${_pair%%"$tab"*}; _dst=${_pair#*:}
      in_list "$_dst" "$_dsts" || block "dry run shows an update of $_dst, which was not requested"
      is_protected "${_dst#refs/heads/}" && block "dry run shows a push to protected $_dst"
      case $_flag in
        -) block "dry run shows deletion of $_dst" ;;
        +) [ "$_lease" = 1 ] || block "dry run shows a forced update of $_dst without --force-with-lease" ;;
        '!') _rejected=$((_rejected + 1)) ;;
        ' '|'*'|'=') ;;
        *) block "dry run shows unexpected status '$_flag' for $_dst" ;;
      esac ;;
  esac
done <<EOF
$_out
EOF
[ "$_refs" -gt 0 ] || block "git push --dry-run showed no ref updates (failed or timed out); run it yourself in a terminal to see why"
[ "$_tos" -gt 0 ] || block "dry run output had ref updates but no destination URL"
# A failed dry run is fine only when git finished and rejected a ref
# (e.g. non-fast-forward): the real push will be rejected the same way.
[ "$_rc" = 0 ] || { [ "$_rejected" -gt 0 ] && [ "$_done" = 1 ]; } ||
  block "git push --dry-run failed (status $_rc); run it yourself in a terminal to see why"
exit 0
