#!/bin/sh
# test-push-guard.sh (v2): builds scratch repos with LOCAL bare remotes and
# checks that push-guard.sh allows (exit 0) or blocks (exit 2) each case.
# Then, end to end, it really runs every allowed network command (push,
# fetch, pull, ls-remote) against the local bare remotes and asserts that no
# protected ref (main, backup/*, tags) moved on either remote.
#
# Usage: sh test-push-guard.sh [--shell SH] [--hook PATH]
#            [--json auto|jq|python3|plutil] [--workdir DIR] [--sabotage]
#   --shell     shell used to run the hook (default: sh)
#   --sabotage  run against a stub hook that always exits 0; the run succeeds
#               only if EVERY block case fails (proves the tests can fail).
# Exit status 0 = all expectations met.
# v1 cases are kept; reclassified ones carry a "# v1 ... -> v2 ..." comment.

# shellcheck disable=SC2016
# SC2016: test commands are literal strings; $ and backticks must not expand.
set -u
set -f
here=$(cd "$(dirname -- "$0")" && pwd -P)
HOOK=$here/push-guard.sh
SH='sh'
JSONP=auto
SABOTAGE=0
WORK=${TMPDIR:-/tmp}/push-guard-test.$$
while [ $# -gt 0 ]; do
  case $1 in
    --shell) SH=$2; shift 2 ;;
    --hook) HOOK=$2; shift 2 ;;
    --json) JSONP=$2; shift 2 ;;
    --workdir) WORK=$2; shift 2 ;;
    --sabotage) SABOTAGE=1; shift ;;
    *) echo "unknown arg $1" >&2; exit 64 ;;
  esac
done

# Only ever delete a directory this script created (it holds a marker file).
if [ -e "$WORK" ] && [ ! -f "$WORK/.push-guard-test" ]; then
  echo "refusing to use existing $WORK (not created by this script)" >&2; exit 64
fi
rm -rf "$WORK"
mkdir -p "$WORK" && : >"$WORK/.push-guard-test" || exit 70
T=$(cd "$WORK" && pwd -P)
SHP=$(command -v "$SH") || { echo "shell not found: $SH" >&2; exit 64; }

if [ "$SABOTAGE" = 1 ]; then
  printf '#!/bin/sh\ncat >/dev/null\nexit 0\n' >"$T/stub.sh"
  HOOK=$T/stub.sh
fi

# ------------------------------------------------------------ fixtures ---
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@example.invalid
export GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@example.invalid
export GIT_AUTHOR_DATE='2026-01-01T00:00:00Z' GIT_COMMITTER_DATE='2026-01-01T00:00:00Z'
g() { git "$@" >/dev/null 2>&1 || { echo "fixture failed: git $*" >&2; exit 70; }; }

R=$T/remotes
mkdir -p "$R"
g init -q --bare -b main "$R/origin.git"
g init -q --bare -b main "$R/upstream.git"

# seed both remotes with main and feat
g init -q -b main "$T/seed"
g -C "$T/seed" commit -q --allow-empty -m base
g -C "$T/seed" branch feat
g -C "$T/seed" branch feat2
g -C "$T/seed" branch backup/x
for r in origin upstream; do
  g -C "$T/seed" push -q "$R/$r.git" main feat feat2 backup/x
done

mkclone() { # name -> clone with origin + upstream, main ahead of origin
  g clone -q "$R/origin.git" "$T/$1"
  g -C "$T/$1" remote add upstream "$R/upstream.git"
  g -C "$T/$1" fetch -q upstream
  g -C "$T/$1" commit -q --allow-empty -m "local main work"
  for b in feat feat2 backup/x; do g -C "$T/$1" branch "$b" "origin/$b"; done
  g -C "$T/$1" checkout -q feat
  g -C "$T/$1" commit -q --allow-empty -m "feat work"
  g -C "$T/$1" tag v1
}

mkclone allowed
# feat2 diverged from origin/feat2 (needs a forced update)
g -C "$T/allowed" checkout -q feat2
g -C "$T/allowed" commit -q --allow-empty -m "rewritten"
g -C "$T/allowed" reset -q --hard HEAD~1
g -C "$T/allowed" commit -q --allow-empty -m "rewritten-2"
g -C "$T/allowed" checkout -q feat
# make origin/feat2 something feat2 does not contain
g -C "$T/seed" checkout -q feat2
g -C "$T/seed" commit -q --allow-empty -m "remote-only"
g -C "$T/seed" push -q "$R/origin.git" feat2
g -C "$T/allowed" fetch -q origin
g -C "$T/allowed" branch newfeat
g -C "$T/allowed" worktree add -q -b wtfeat "$T/wt-allowed"
g -C "$T/wt-allowed" commit -q --allow-empty -m "wt work"

mkclone onmain;   g -C "$T/onmain" checkout -q main
mkclone matching; g -C "$T/matching" config push.default matching
mkclone tracking
g -C "$T/tracking" checkout -q -b work origin/main
g -C "$T/tracking" commit -q --allow-empty -m "tracking work"
g -C "$T/tracking" config push.default upstream
mkclone tracksimple
g -C "$T/tracksimple" checkout -q -b work origin/main
g -C "$T/tracksimple" commit -q --allow-empty -m "tracking work"
mkclone mapped;   g -C "$T/mapped" config remote.origin.push refs/heads/feat:refs/heads/main
mkclone forcecfg
g -C "$T/forcecfg" checkout -q feat2
g -C "$T/forcecfg" reset -q --hard origin/feat2~1
g -C "$T/forcecfg" commit -q --allow-empty -m other
g -C "$T/forcecfg" config remote.origin.push +refs/heads/feat2:refs/heads/feat2
mkclone delcfg;   g -C "$T/delcfg" config remote.origin.push :refs/heads/feat
mkclone mirrorcfg; g -C "$T/mirrorcfg" config remote.origin.mirror true
g -C "$T/mirrorcfg" branch -D backup/x
mkclone pushurl;  g -C "$T/pushurl" config remote.origin.pushurl "$R/upstream.git"
mkclone withsub;  mkdir -p "$T/withsub/.git/modules/x"
# ssh-style URL through a fake ssh that serves the local bare repos
cat >"$T/fakessh" <<EOF
#!/bin/sh
for a in "\$@"; do last=\$a; done
cd "$R" || exit 1
c=\${last#git-}
eval "git \$c"
EOF
chmod +x "$T/fakessh"
mkclone sshish
g -C "$T/sshish" remote set-url origin "git@fakehost:origin.git"
g -C "$T/sshish" remote set-url upstream "git@fakehost:upstream.git"

g -C "$T/allowed" branch mainx
g -C "$T/allowed" branch feat5
g -C "$T/allowed" config alias.p push
g -C "$T/allowed" config alias.sh '!git push origin main'
g -C "$T/allowed" config alias.st status
g -C "$T/allowed" config alias.pp p
g -C "$T/allowed" config alias.l 'log --oneline -1'
g -C "$T/allowed" config alias.cs '-c x.y=z status'
g -C "$T/allowed" config alias.up 'remote update'
mkclone viaurl
g -C "$T/viaurl" config branch.feat.remote "$R/upstream.git"
g -C "$T/viaurl" config branch.feat.merge refs/heads/feat
mkclone pushdef
g -C "$T/pushdef" config remote.pushDefault upstream
# v2 fixtures: a symref branch, a fast-forwardable pull, a worktree whose
# commondir was rewritten (sandbox can write .git/worktrees/*) to another repo
g -C "$T/allowed" symbolic-ref refs/heads/symb refs/heads/main
mkclone ffpull
g -C "$T/ffpull" checkout -q -b behind origin/main
g -C "$T/allowed" worktree add -q -b wttamper "$T/wt-tamper"

mkclone unlisted
# RT2 fixtures: programs that would run if an abbreviated option got through
mkdir -p "$T/rt2/tpl/hooks" "$T/rt2/hp"
printf '#!/bin/sh\necho ran >>%s/rt2/PWNED\nexec git-upload-pack "$@"\n' "$T" >"$T/rt2/evil.sh"
chmod +x "$T/rt2/evil.sh"
g init -q --bare -b main "$T/rt2/bare.git"
# attacker repo reachable as <attacker>/lnk/.. where lnk -> allowed/sub
mkdir -p "$T/allowed/sub"
g init -q -b feat "$T/attacker"
g -C "$T/attacker" commit -q --allow-empty -m evil
g -C "$T/attacker" remote add origin "$R/origin.git"
g -C "$T/attacker" config remote.origin.push +refs/heads/feat:refs/heads/main
ln -s "$T/allowed/sub" "$T/attacker/lnk"
g -C "$T/unlisted" worktree add -q -b wtu "$T/wt-unlisted"
# RT4 fixtures: payload files (the hook never runs them) and a clone whose
# gitlink 'sub' is a plain nested clone (no .git/modules) with its own config
mkdir -p "$T/rt4/fakehome" "$T/rt4/fakebin" "$T/rt4/hp"
printf '[remote "origin"]\n\tpushurl = %s/rt4/evil.git\n' "$T" >"$T/rt4/evil.cfg"
printf '[core] hooksPath = %s/rt4/hp\n' "$T" >"$T/rt4/evil2.cfg"
mkclone nestedsub
g init -q -b main "$T/rt4/subsrc"
g -C "$T/rt4/subsrc" commit -q --allow-empty -m s1
g clone -q "$T/rt4/subsrc" "$T/nestedsub/sub"
printf '[submodule "sub"]\n\tpath = sub\n\turl = ./sub\n' >"$T/nestedsub/.gitmodules"
g -C "$T/nestedsub" add .gitmodules
g -C "$T/nestedsub" update-index --add --cacheinfo "160000,$(git -C "$T/nestedsub/sub" rev-parse HEAD),sub"
g -C "$T/nestedsub" commit -q -m gitlink
g -C "$T/nestedsub" branch feat-sub
g -C "$T/nestedsub/sub" config remote.origin.uploadpack "$T/rt4/up.sh"

mkdir -p "$T/redir" "$T/redir2" "$T/plain"
printf 'gitdir: %s/allowed/.git\n' "$T" >"$T/redir/.git"
printf 'gitdir: %s/allowed/.git/worktrees/wt-allowed\n' "$T" >"$T/redir2/.git"
ln -s "$T/allowed" "$T/allowed-link"

# RT5 fixtures: attacker repos whose pre-commit hook, core.fsmonitor and
# clean filter would run if a git companion executed there. The hook itself
# must never trigger them (checked at the end).
mkdir -p "$T/rt5"
printf '#!/bin/sh\necho ran >>%s/rt5/PWNED\n' "$T" >"$T/rt5/payload.sh"
chmod +x "$T/rt5/payload.sh"
g init -q -b feat "$T/rt5/attacker"
cp "$T/rt5/payload.sh" "$T/rt5/attacker/.git/hooks/pre-commit"
g init -q -b feat "$T/rt5/att2"
g -C "$T/rt5/att2" config core.fsmonitor "$T/rt5/payload.sh"
g init -q -b feat "$T/rt5/att3"
printf '* filter=x\n' >"$T/rt5/att3/.gitattributes"
printf 'data\n' >"$T/rt5/att3/f"
g -C "$T/rt5/att3" config filter.x.clean "$T/rt5/payload.sh"

# FRT2 fixtures: linked worktrees whose top level is a git dir (or contains an
# allowed repo). 'evil' adds hooks/post-merge; pulling it into a worktree
# rooted at <repo>/.git would plant and run a hook unsandboxed. A separate
# allowed clone keeps the plain "$A/.git" cases meaning what they say.
mkclone frt2
AA=$T/frt2
printf '#!/bin/sh\necho PWNED-unsandboxed >>%s/frt2-PWNED\n' "$T" >"$T/frt2-payload"
_e=$(git -C "$AA" mktree </dev/null)
_c0=$(git -C "$AA" commit-tree -m wbase "$_e")
_b=$(git -C "$AA" hash-object -w "$T/frt2-payload")
_h=$(printf '100755 blob %s\tpost-merge\n' "$_b" | git -C "$AA" mktree)
_t=$(printf '040000 tree %s\thooks\n' "$_h" | git -C "$AA" mktree)
_c1=$(git -C "$AA" commit-tree -p "$_c0" -m evil "$_t")
g -C "$AA" update-ref refs/heads/wbase "$_c0"
g -C "$AA" update-ref refs/heads/evil "$_c1"
g -C "$AA" push -q origin evil:refs/heads/evil
mkwt() { # repo git-dir-name toplevel head: hand-written linked worktree
  mkdir -p "$1/.git/worktrees/$2" "$3"
  echo "ref: refs/heads/$4" >"$1/.git/worktrees/$2/HEAD"
  echo '../..' >"$1/.git/worktrees/$2/commondir"
  echo "$3/.git" >"$1/.git/worktrees/$2/gitdir"
  echo "gitdir: $1/.git/worktrees/$2" >"$3/.git"
}
mkwt "$AA" frtgd "$AA/.git" wbase          # top level = the common dir itself
mkwt "$AA" frtrefs "$AA/.git/refs" wbase   # top level inside the common dir
mkwt "$AA" frtwt "$AA/.git/worktrees/frtgd/x" wbase  # inside a worktree's git dir
mkwt "$AA" frtx "$T/allowed/.git/frtx" wbase     # inside ANOTHER allowed repo's .git
mkdir -p "$T/outer"
mkclone outer/inner
mkwt "$T/outer/inner" frtouter "$T/outer" main   # top level contains an allowed repo
AA=$T/allowed
# a legitimate agent worktree inside the work tree
g -C "$AA" worktree add -q -b agentwt "$AA/.claude/worktrees/agent1"
g -C "$AA/.claude/worktrees/agent1" commit -q --allow-empty -m "agent work"

export GIT_SSH="$T/fakessh" GIT_SSH_VARIANT=simple

HOOKARGS="--remote origin --protect main --protect backup/* --allow-force-with-lease"
for r in allowed onmain matching tracking tracksimple mapped forcecfg delcfg mirrorcfg pushurl withsub sshish viaurl pushdef nestedsub ffpull frt2 outer/inner; do
  HOOKARGS="$HOOKARGS --repo $T/$r"
done
export CLAUDE_PROJECT_DIR="$T/allowed"
export PUSH_GUARD_JSON="$JSONP"

# --------------------------------------------------------------- runner ---
json_str() {
  printf '%s\n' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e "s/$(printf '\t')/\\\\t/g" |
    awk 'NR>1{printf "\\n"} {printf "%s", $0}'
}

pass=0 fail=0 nblock=0 blockfail=0
run() { # expect(allow|block) cwd command [extra hook args...]
  exp=$1 dir=$2 c=$3; shift 3
  e2e_dir=$dir
  j=$(printf '{"session_id":"t","hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"%s","description":"x"},"cwd":"%s","permission_mode":"auto"}' \
      "$(json_str "$c")" "$(json_str "$dir")")
  rawrun "$exp" "$c" "$j" "$@"
  e2e_dir=''
}
rawrun() { # expect label json [extra hook args]
  exp=$1 label=$2 j=$3; shift 3
  # shellcheck disable=SC2086
  err=$(printf '%s' "$j" | PATH=${HOOKPATH:-$PATH} "$SHP" "$HOOK" $HOOKARGS "$@" 2>&1 >/dev/null)
  rc=$?
  case $exp in allow) want=0 ;; block) want=2; nblock=$((nblock + 1)) ;; esac
  if [ "$rc" = "$want" ]; then
    pass=$((pass + 1))
    # remember allowed network commands for the end-to-end pass
    if [ "$exp" = allow ] && [ -n "${e2e_dir:-}" ]; then
      case $label in
        'git push '*|'git fetch '*|'git pull '*|'git ls-remote '*) printf '%s\t%s\t%s\n' "$e2e_dir" "$label" "$*" >>"$T/e2e.list" ;;
      esac
    fi
    [ -n "${VERBOSE:-}" ] && printf 'ok   %-5s %s\n' "$exp" "$label"
    [ -n "${VERBOSE:-}" ] && [ -n "$err" ] && printf '       %s\n' "$(printf '%s' "$err" | head -1)"
  else
    fail=$((fail + 1))
    [ "$exp" = block ] && blockfail=$((blockfail + 1))
    if [ "$SABOTAGE" = 0 ]; then
      printf 'FAIL want %-5s got rc=%s  %s\n' "$exp" "$rc" "$label"
      [ -n "$err" ] && printf '      %s\n' "$(printf '%s' "$err" | head -1)"
    fi
  fi
}

A=$T/allowed
# --- plain pushes
run allow "$A" 'git push origin feat:refs/heads/feat'
run allow "$A" 'git push -u origin feat:refs/heads/feat'
run allow "$A" 'git push origin feat:refs/heads/feat -u'
run allow "$A" 'git push --set-upstream origin newfeat:refs/heads/newfeat'
run block "$A" 'git push -uv origin feat:refs/heads/feat'  # v1 allow -> v2 block
run allow "$A" 'git push origin feat:refs/heads/feat newfeat:refs/heads/newfeat'
run block "$A" 'git push origin refs/heads/feat:refs/heads/feat'  # v1 allow -> v2 block
run block "$A" 'git push origin HEAD:refs/heads/feat'  # v1 allow -> v2 block
run allow "$A" 'git push --dry-run origin feat:refs/heads/feat'
run block "$A" 'git push origin -- feat:refs/heads/feat'  # v1 allow -> v2 block
run allow "$A" 'git push --force-with-lease origin feat2:refs/heads/feat2'
run allow "$A" 'git push --force-with-lease --force-if-includes origin feat2:refs/heads/feat2'
run block "$A" 'git push --force-with-lease origin feat2:refs/heads/feat2' --protect feat2
run allow "$A" 'git push origin feat2:refs/heads/feat2'   # non-fast-forward: git rejects it, nothing forced
# lease not allowed when the hook flag is absent: use a separate arg list
HOOKARGS_SAVE=$HOOKARGS
HOOKARGS=$(printf '%s' "$HOOKARGS" | sed 's/ --allow-force-with-lease//')
run block "$A" 'git push --force-with-lease origin feat2:refs/heads/feat2'
HOOKARGS=$HOOKARGS_SAVE
run block "$A" 'git push -f origin feat:refs/heads/feat'
run block "$A" 'git push --force origin feat:refs/heads/feat'
run block "$A" 'git push origin feat:refs/heads/feat --force'
run block "$A" 'git push --no-verify origin feat:refs/heads/feat'
run block "$A" 'git push --repo=../remotes/upstream.git origin feat:refs/heads/feat'
run block "$A" 'git push --recurse-submodules=on-demand origin feat:refs/heads/feat'
run block "$A" 'git push -o ci.skip origin feat:refs/heads/feat'
# --- protected destinations
run block "$A" 'git push origin main'
run block "$A" 'git push origin HEAD:main'
run block "$A" 'git push origin feat:main'
run block "$A" 'git push origin feat:feat3'
run block "$A" 'git push origin +feat'
run block "$A" 'git push origin +main'
run block "$A" 'git push origin refs/heads/main'
run block "$A" 'git push origin heads/main'
run block "$A" 'git push origin backup/x'
run block "$A" 'git push origin refs/heads/backup/x'
run block "$A" 'git push origin feat:refs/heads/feat main'
run block "$A" 'git push origin refs/remotes/origin/feat'
run block "$A" 'git push origin feat:HEAD'
run block "$A" 'git push --all origin'
run block "$A" 'git push --mirror origin'
run block "$A" 'git push --prune origin'
run block "$A" 'git push --tags origin'
run block "$A" 'git push --follow-tags origin feat:refs/heads/feat'
run block "$A" 'git push origin v1'
run block "$A" 'git push origin refs/tags/v1'
run block "$A" 'git push --delete origin feat:refs/heads/feat'
run block "$A" 'git push -d origin feat:refs/heads/feat'
run block "$A" 'git push origin :feat'
run block "$A" 'git push origin --delete feat:refs/heads/feat'
# --- remotes
run block "$A" 'git push upstream feat:refs/heads/feat'
run block "$A" "git push $T/remotes/upstream.git feat:refs/heads/feat"
run block "$A" "git push $T/remotes/origin.git feat:refs/heads/feat"
run block "$A" 'git push ../remotes/origin.git feat:refs/heads/feat'
run block "$A" 'git push git@github.com:lanl/shoccs.git feat:refs/heads/feat'
run block "$A" 'git push https://github.com/lanl/shoccs.git feat:refs/heads/feat'
run block "$A" 'git push nosuchremote feat:refs/heads/feat'
run block "$T/pushurl" 'git push origin feat:refs/heads/feat'
run allow "$T/sshish" 'git push origin feat:refs/heads/feat'
run block "$T/sshish" 'git push upstream feat:refs/heads/feat'
run block "$T/sshish" 'git push git@fakehost:origin.git feat:refs/heads/feat'
# --- dangerous git options
run block "$A" 'git -c core.sshCommand=evil push origin feat:refs/heads/feat'
run block "$A" 'git -c protocol.ext.allow=always push origin feat:refs/heads/feat'
run block "$A" 'git --config-env=core.sshCommand=X push origin feat:refs/heads/feat'
run block "$A" 'git --exec-path=/tmp push origin feat:refs/heads/feat'
run block "$A" "git --git-dir=$T/unlisted/.git push origin feat:refs/heads/feat"
run block "$A" "git --work-tree=$T/plain push origin feat:refs/heads/feat"
run block "$A" 'git --namespace=x push origin feat:refs/heads/feat'
run block "$A" 'git push --receive-pack=evil origin feat:refs/heads/feat'
run block "$A" 'git push --exec=evil origin feat:refs/heads/feat'
run block "$A" 'git fetch --upload-pack=evil origin'
run block "$A" 'git fetch -u origin'
run block "$A" 'git ls-remote --upload-pack=evil origin'
run block "$A" 'git ls-remote -u evil origin'
run block "$A" 'git pull --upload-pack=evil origin feat'
run block "$A" 'git send-pack ../remotes/origin.git main'
run block "$A" 'git-send-pack ../remotes/origin.git main'
run block "$A" 'git-receive-pack ../remotes/origin.git'
run block "$A" 'git archive --remote=../remotes/origin.git main'
run allow "$A" 'git -c color.ui=never log --oneline -1'
run allow "$A" 'git --no-pager log -1'
# --- env prefixes and wrappers
run block "$A" 'GIT_SSH_COMMAND=evil git push origin feat:refs/heads/feat'
run block "$A" 'HOME=/tmp git push origin feat:refs/heads/feat'
run block "$A" 'GIT_DIR=/x git fetch origin'
run block "$A" 'env git push origin feat:refs/heads/feat'
run block "$A" 'env -i git push origin feat:refs/heads/feat'
run block "$A" 'command git push origin feat:refs/heads/feat'
run block "$A" 'exec git push origin feat:refs/heads/feat'
run block "$A" 'time git push origin feat:refs/heads/feat'
run block "$A" 'nohup git push origin feat:refs/heads/feat'
run block "$A" 'xargs git push origin < /dev/null'
run block "$A" "sh -c 'git push origin main'"
run block "$A" 'bash -c "git push origin main"'
run block "$A" '(git push origin feat)'
run block "$A" '{ git push origin feat:refs/heads/feat; }'
run block "$A" 'if git push origin feat:refs/heads/feat; then echo ok; fi'
run block "$A" 'for b in main; do git push origin $b; done'
run block "$A" 'export GIT_SSH_COMMAND=evil; git push origin feat:refs/heads/feat'
run block "$A" 'GIT_SSH_COMMAND=evil; git push origin feat:refs/heads/feat'
run block "$A" 'PATH=/tmp/evil:$PATH && git push origin feat:refs/heads/feat'
run block "$A" 'alias git=/tmp/evil; git push origin feat:refs/heads/feat'
run block "$A" 'git() { /tmp/evil; }; git push origin feat:refs/heads/feat'
run block "$A" 'eval x; git push origin feat:refs/heads/feat'
run block "$A" '. ./evil.sh && git push origin feat:refs/heads/feat'
run block "$A" 'hash -p /tmp/evil git; git push origin feat:refs/heads/feat'
run block "$A" 'echo git push origin main'
# --- obfuscation
run block "$A" "git push origin ma''in"
run block "$A" 'git push origin "main"'
run block "$A" "git push origin 'feat'"
run block "$A" 'git push origin m\ain'
run block "$A" 'git push origin $(echo main)'
run block "$A" 'git push origin `echo main`'
run block "$A" 'git push origin ${B}'
run block "$A" 'git push origin ma*'
run block "$A" 'git push origin ma?n'
run block "$A" 'git push origin ma[i]n'
run block "$A" "g''it push origin main"
run block "$A" '"git" push origin main'
run block "$A" 'g\it push origin main'
run block "$A" '"g"it push origin main'
run block "$A" "gi''t push origin main"
run block "$A" '\git push origin main'
run block "$A" 'GIT push origin main'
run block "$A" "git 'push' origin main"
run block "$A" 'git pu\sh origin main'
run block "$A" '/usr/bin/git push origin main'
run block "$A" '/usr/bin/git push origin feat:refs/heads/feat'  # v1 allow -> v2 block
run block "$A" 'git push origin feat:refs/heads/feat > /tmp/out'
run block "$A" 'git push origin feat:refs/heads/feat <<EOF'
run block "$A" 'git	push	origin	main'
run block "$A" "$(printf 'git push origin main\r')"
run block "$A" 'git push origin feat~0:main'
# --- chains
run block "$A" 'echo ok && git push origin main'
run block "$A" 'echo ok && git push origin feat:refs/heads/feat'  # v1 allow -> v2 block
run block "$A" 'git push origin feat:refs/heads/feat; git push origin main'
run block "$A" 'git push origin feat:refs/heads/feat && git push origin main'
run block "$A" 'git push origin feat:refs/heads/feat || git push origin main'
run block "$A" 'git push origin main | cat'
run block "$A" 'git push origin feat:refs/heads/feat & git push origin main'
run block "$A" 'git push origin feat:refs/heads/feat |& git push origin main'
run block "$A" "$(printf 'echo hi\ngit push origin main')"
run block "$A" 'git push origin feat:refs/heads/feat 2>&1 | tail -5'  # v1 allow -> v2 block
run block "$A" 'git push origin feat:refs/heads/feat >/dev/null 2>&1'  # v1 allow -> v2 block
run block "$A" 'git fetch --no-recurse-submodules origin 2>/dev/null'  # v1 allow -> v2 block
# RT5: no local git command shares a line with a network command
run block "$A" 'git status && git push origin feat:refs/heads/feat'
# RT4: quotes anywhere on a pushing line are refused (run the commit separately)
run block "$A" 'git add -A && git commit -m "push the fix to origin main" && git push origin feat:refs/heads/feat'
run block "$A" 'git commit -m "msg; git push origin main"'
# --- cd / -C
run block "$A" "cd $T/unlisted && git push origin feat:refs/heads/feat"
run block "$A" 'cd ../unlisted && git push origin feat:refs/heads/feat'
run block "$T/unlisted" "cd $T/allowed && git push origin feat:refs/heads/feat"  # v1 allow -> v2 block
run block "$T" 'cd allowed && git push origin feat:refs/heads/feat'  # v1 allow -> v2 block
run block "$T" "cd $T/allowed; git push origin feat:refs/heads/feat"  # v1 allow -> v2 block
run block "$T" 'cd allowed && cd ../unlisted && git push origin feat:refs/heads/feat'
run block "$A" "git -C $T/unlisted push origin feat:refs/heads/feat"
run block "$T/unlisted" "git -C $T/allowed push origin feat:refs/heads/feat"  # v1 allow -> v2 block
run block "$T" 'git -C allowed push origin feat:refs/heads/feat'  # v1 allow -> v2 block
run block "$T" 'git -C allowed -C ../unlisted push origin feat:refs/heads/feat'
run block "$A" 'cd /nonexistent; git push origin feat:refs/heads/feat'
run block "$A" 'cd - && git push origin feat:refs/heads/feat'
run block "$A" "cd \"$T/allowed\" && git push origin feat"
run block "$A" 'popd; git push origin feat:refs/heads/feat'
run block "$A" "{ cd $T/unlisted; }; git push origin feat:refs/heads/feat"
run block "$T" 'cd allowed-link && git push origin feat:refs/heads/feat'  # v1 allow -> v2 block
# --- implicit destinations (dry run)
run block "$A" 'git push'
run block "$A" 'git push origin'
run block "$A" 'git push origin feat'
run block "$A" 'git push -u origin feat'
run block "$A" 'git push origin HEAD'
run block "$T/onmain" 'git push'
run block "$T/onmain" 'git push origin'
run block "$T/onmain" 'git push origin HEAD'
run block "$T/onmain" 'git push origin @'
run block "$T/tracking" 'git push'
run block "$T/tracking" 'git push origin'
run block "$T/tracksimple" 'git push'
run block "$T/matching" 'git push'
run block "$T/matching" 'git push origin'
run block "$T/mapped" 'git push origin'
run block "$T/mapped" 'git push origin feat'
run block "$T/forcecfg" 'git push'
run block "$T/forcecfg" 'git push origin feat2'
run block "$T/delcfg" 'git push origin'
run block "$T/mirrorcfg" 'git push origin'
# --- fetch / pull / ls-remote / clone
run allow "$A" 'git fetch --no-recurse-submodules'
run allow "$A" 'git fetch --no-recurse-submodules origin'
run allow "$A" 'git fetch --no-recurse-submodules upstream'
run allow "$A" 'git fetch --no-recurse-submodules --prune origin'
run allow "$A" 'git fetch --no-recurse-submodules --all'
run block "$A" 'git fetch --no-recurse-submodules --multiple origin upstream'  # v1 allow -> v2 block
run block "$A" 'git fetch --no-recurse-submodules --depth 1 origin main'  # v1 allow -> v2 block
run block "$A" 'git fetch --no-recurse-submodules origin main:refs/remotes/origin/main'  # v1 allow -> v2 block
run block "$A" 'git fetch --multiple origin ../remotes/upstream.git'
run block "$A" 'git fetch ../remotes/upstream.git'
run block "$A" "git fetch $T/remotes/origin.git"
run block "$A" 'git fetch https://example.com/x.git'
run block "$A" 'git fetch git@github.com:lanl/shoccs.git main'
run block "$A" 'git fetch ext::sh'
run block "$A" 'git fetch .'
run block "$A" 'git pull --no-recurse-submodules origin feat'  # v1 allow -> v2 block
run block "$A" 'git pull --no-recurse-submodules --rebase origin feat'  # v1 allow -> v2 block
run block "$A" 'git pull --no-recurse-submodules'  # v1 allow -> v2 block
run block "$A" 'git pull https://example.com/x.git main'
run allow "$A" 'git ls-remote origin'
run allow "$A" 'git ls-remote --heads upstream'
run block "$A" 'git ls-remote https://example.com/x.git'
run block "$A" 'git ls-remote ../remotes/origin.git'
run block "$T/withsub" 'git fetch origin'
run allow "$T/withsub" 'git fetch --no-recurse-submodules origin'
run block "$T/withsub" 'git pull origin feat'
run allow "$T/wt-allowed" 'git fetch --no-recurse-submodules origin'
run block "$T/unlisted" 'git fetch origin'
run block "$T" 'git clone https://github.com/pbrady/shoccs.git x'  # v1 allow -> v2 block
run block "$T" 'git clone --depth 1 https://github.com/pbrady/shoccs.git'  # v1 allow -> v2 block
run block "$T" 'git clone --upload-pack=evil https://github.com/pbrady/shoccs.git'
run block "$T" 'git clone -u evil https://github.com/pbrady/shoccs.git'
run block "$T" 'git clone -c core.sshCommand=evil https://github.com/pbrady/shoccs.git'
run block "$T" 'git clone --config core.sshCommand=evil https://github.com/pbrady/shoccs.git'
run block "$T" 'git clone --template=/tmp/t https://github.com/pbrady/shoccs.git'
run block "$T" 'git -c core.sshCommand=evil clone https://github.com/pbrady/shoccs.git'
run block "$T" "git clone 'ext::sh -c evil' x"
run block "$T" 'git clone ext::sh x'
# --- repository allowlist
run block "$T/unlisted" 'git push origin feat:refs/heads/feat'
run block "$T/unlisted" 'git push origin wtu:refs/heads/wtu'
run allow "$T/wt-allowed" 'git push origin wtfeat:refs/heads/wtfeat'
run allow "$T/wt-allowed" 'git push -u origin wtfeat:refs/heads/wtfeat'
run block "$T/wt-allowed" 'git push origin main'
run block "$T/wt-unlisted" 'git push origin wtu:refs/heads/wtu'
run block "$T/redir" 'git push origin feat:refs/heads/feat'
run block "$T/redir" 'git fetch origin'
run block "$T/redir2" 'git push origin wtfeat:refs/heads/wtfeat'
run block "$T/plain" 'git push origin feat:refs/heads/feat'
run block "$A/.git" 'git push origin feat:refs/heads/feat'
# project dir only counts with --allow-project-dir
HOOKARGS_SAVE=$HOOKARGS
HOOKARGS='--remote origin --protect main --protect backup/*'
run block "$A" 'git push origin feat:refs/heads/feat'
run block "$A" 'git push origin feat:refs/heads/feat' --allow-project-dir  # v1 allow -> v2 block
run block "$T/unlisted" 'git push origin feat:refs/heads/feat' --allow-project-dir
HOOKARGS=$HOOKARGS_SAVE
# --- non-git commands and harmless git
run allow "$A" 'ls -la'
run allow "$A" 'echo "hello $(date)" | tr a b > /tmp/x'
run allow "$A" 'cd scripts/stencil_gen && SYMPY_CACHE_SIZE=50000 uv run pytest tests/ -x -q'
run allow "$A" 'grep -rn push src'
run allow "$A" 'git status'
run allow "$A" 'git log --oneline | grep push'
run allow "$A" 'git diff HEAD~1 -- "src/a b.cpp"'
run allow "$A" 'git commit -m "$(cat <<EOF
fix push path
EOF
)"'
run allow "$A" 'export FOO=1; git status'
run allow "$A" 'git remote -v'
run allow "$A" 'ls .git'
# --- review round 2: redirect-fusion, command substitution, aliases, implicit remotes
run block "$A" 'git push origin feat:refs/heads/feat main>&1x'
run block "$A" 'git push origin feat:refs/heads/feat main2>&1x'
run block "$A" 'git push origin feat:refs/heads/feat main>/dev/nullx'
run block "$A" 'git push origin feat:refs/heads/feat main</dev/nullx'
run block "$A" 'git push origin feat:refs/heads/feat main&>/dev/nullx'
run block "$A" 'git push origin feat:refs/heads/feat 2>&1|tail -5'  # v1 allow -> v2 block
run block "$A" 'git push origin feat:refs/heads/feat 2>&1; echo done'  # v1 allow -> v2 block
run block "$A" '2>&1 git push origin feat:refs/heads/feat'  # v1 allow -> v2 block
run block "$A" 'x=$(git push origin main)'
run block "$A" 'echo $(git push origin main)'
run block "$A" 'echo `git push origin main`'
run block "$A" 'cat <(git push origin main)'
run allow "$A" 'git p origin main'  # v1 block -> v2 allow: no network verb, runs sandboxed
run allow "$A" 'git p origin feat'  # v1 block -> v2 allow: no network verb, runs sandboxed
run allow "$A" 'git sh'  # v1 block -> v2 allow: no network verb, runs sandboxed
run allow "$A" 'git st'
run block "$T/viaurl" 'git push'
run block "$T/viaurl" 'git fetch --no-recurse-submodules'
run block "$T/viaurl" 'git pull --no-recurse-submodules'
run block "$T/viaurl" 'git pull --no-recurse-submodules --all'
run allow "$T/viaurl" 'git fetch --no-recurse-submodules --all'
run block "$T/pushdef" 'git push'
run allow "$T/pushdef" 'git push origin feat:refs/heads/feat'
run block "$A" 'git push origin feat:refs/heads/feat Main'
run block "$A" 'gh pr checkout 1 && git push'
# --- review round 3: simpler stricter parsing
# redirects: only whole-word allowlisted forms are ignored
run block "$A" 'git push origin feat:refs/heads/feat >>/dev/null'  # v1 allow -> v2 block
run block "$A" 'git push origin feat:refs/heads/feat > /dev/null 2>&1 &'  # v1 allow -> v2 block
run block "$A" 'git push origin feat:refs/heads/feat 2>/dev/null; echo done'  # v1 allow -> v2 block
run block "$A" 'git push origin feat:refs/heads/feat 2>&1>/dev/null'
run block "$A" 'git push origin feat2>&1x'
run block "$A" 'git push origin feat:refs/heads/feat main>>/dev/nullx'
run block "$A" "$(printf 'git push origin feat\t2>&1')"
# ( ) and backticks anywhere block a network command
run block "$A" 'echo $(date) && git push origin feat:refs/heads/feat'
run block "$A" "(cd $A && git push origin feat)"
run block "$A" 'x=`git push origin feat`'
run block "$A" 'git commit -m "$(cat <<EOF
msg
EOF
)" && git push origin feat'
# every other segment must be a known-harmless companion
run block "$A" 'true && git push origin feat:refs/heads/feat'  # v1 allow -> v2 block
run block "$A" 'sleep 1; git fetch --no-recurse-submodules origin'  # v1 allow -> v2 block
run block "$T" 'cd allowed && git fetch --no-recurse-submodules'  # v1 allow -> v2 block
run block "$A" 'git checkout main && git push origin HEAD'
run block "$A" 'git status; git switch main; git push origin HEAD'
run block "$A" 'git tag feat5 && git push origin feat5:refs/heads/feat5'
run block "$A" 'git push origin feat:refs/heads/feat; git config remote.origin.pushurl x'
run block "$A" 'B=main; git push origin feat:refs/heads/feat'
run block "$A" 'git push origin feat:refs/heads/feat; then'
# aliases and external subcommands
run allow "$A" 'git l'
run allow "$A" 'git nosuchcmd'
run allow "$A" 'git pp origin feat'  # v1 block -> v2 allow: no network verb, runs sandboxed
run allow "$A" 'git cs'  # v1 block -> v2 allow: no network verb, runs sandboxed
run allow "$A" 'git up'  # v1 block -> v2 allow: no network verb, runs sandboxed
run block "$A" 'git -c alias.zz=push zz origin main'
run block "$A" 'GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=alias.zz GIT_CONFIG_VALUE_0=push git zz origin main'
run allow "$A" 'export GIT_CONFIG_COUNT=1; git zz origin main'  # v1 block -> v2 allow: no network verb, runs sandboxed
run allow "$A" 'env git zz origin main'  # v1 block -> v2 allow: no network verb, runs sandboxed
run allow "$A" 'cd - && git l'  # v1 block -> v2 allow: no network verb, runs sandboxed
run block "$A" 'git PUSH origin main'
run block "$A" 'git remote-ext x y'
run block "$A" 'git daemon --export-all'
run block "$A" 'git-remote-ext x y'
# --- review round 4 (RT2): exact option allowlists, clone source, cd '..', explicit destinations
E=$T/rt2/evil.sh
run block "$A" "git fetch --upload-p=$E origin"
run block "$A" "git ls-remote --upload-p=$E origin"
run block "$A" "git pull --upload-p=$E origin feat"
run block "$A" "git pull --upload-pa=$E origin feat"
run block "$T" "git clone --upload-p=$E $T/rt2/bare.git $T/rt2/c1"
run block "$T" "git clone --templ=$T/rt2/tpl $T/rt2/bare.git $T/rt2/c3"
run block "$T" "git clone --conf=core.hooksPath=$T/rt2/hp $T/rt2/bare.git $T/rt2/c5"
run block "$T" "git clone $T/rt2/bare.git $T/rt2/c3"
run block "$T" 'git clone file:///tmp/x.git'
run block "$T" 'git clone git@github.com:pbrady/shoccs.git'
run block "$T" 'git clone https://example.com/x.git'
run block "$T" 'git clone https://github.com/pbrady/shoccs.git a b'
run block "$T" 'git clone --recurse-submodules https://github.com/pbrady/shoccs.git'
run block "$T" 'git clone --reference /tmp/x https://github.com/pbrady/shoccs.git'
run block "$T" 'git clone --separate-git-dir=/tmp/x https://github.com/pbrady/shoccs.git'
run block "$T" 'git clone --depth'
run block "$T" 'git clone -q --depth=1 --branch main --single-branch https://github.com/pbrady/shoccs.git x'  # v1 allow -> v2 block
run block "$A" 'git fetch --upload origin'
run block "$A" 'git fetch --exe=x origin'
run block "$A" 'git fetch --recurse-sub origin'
run block "$A" 'git fetch --recurse-submodules=yes origin'
run block "$A" 'git fetch --prun origin'
run block "$A" 'git fetch --end-of-options origin'
run block "$A" 'git fetch -pu origin'
run block "$A" 'git fetch -o x origin'
run block "$A" 'git fetch --set-upstream origin feat'
run block "$A" 'git fetch --depth'
run block "$A" 'git fetch --no-recurse-submodules -pv origin'  # v1 allow -> v2 block
run block "$A" 'git fetch --no-recurse-submodules --depth=1 --jobs 2 origin'  # v1 allow -> v2 block
run block "$A" 'git fetch --no-recurse-submodules -j4 --no-tags origin'  # v1 allow -> v2 block
run block "$A" 'git pull -s evil origin feat'
run block "$A" 'git pull --strategy=evil origin feat'
run block "$A" 'git pull -S origin feat'
run block "$A" 'git pull --rebase=interactive origin feat'
run allow "$A" 'git pull --no-recurse-submodules --ff-only -q origin feat'
run block "$A" 'git pull --no-recurse-submodules -Xours --no-edit origin feat'  # v1 allow -> v2 block
run block "$A" 'git ls-remote --exec=x origin'
run block "$A" 'git ls-remote --upload-pack x origin'
run block "$A" 'git ls-remote -ht --refs origin'  # v1 allow -> v2 block
run block "$A" 'git ls-remote --sort version:refname --tags origin'  # v1 allow -> v2 block
# cd/pushd through a symlink then '..': the shell lands in the attacker repo
run block "$T" "cd $T/attacker/lnk/.. && git push origin feat:refs/heads/feat"
run block "$T" "cd $T/attacker/lnk && cd .. && git push origin feat:refs/heads/feat"
run block "$T" "pushd $T/attacker/lnk/.. && git push origin feat:refs/heads/feat"
run block "$T/attacker/lnk" 'cd .. && git push origin feat:refs/heads/feat'
run block "$T/attacker/lnk" 'cd ../lnk/.. && git fetch origin'
run block "$A" 'cd .. && cd allowed && git push origin feat:refs/heads/feat'
run block "$A" 'pushd && git push origin feat:refs/heads/feat'
run block "$A" 'pushd +1 && git push origin feat:refs/heads/feat'
run block "$T" "pushd $T/allowed && git push origin feat:refs/heads/feat"  # v1 allow -> v2 block
run block "$A" 'cd ./sub && git -C .. push origin feat:refs/heads/feat'  # v1 allow -> v2 block
# explicit destinations (symref race: an implicit destination is resolved
# from local refs after the hook's dry run)
run block "$A" 'git push origin symb'
run block "$A" 'git push origin feat:main'
run block "$A" 'git push origin feat:heads/feat'
run block "$A" 'git push origin feat:refs/heads/main'
run block "$A" 'git push origin main:refs/heads/main'
run block "$A" 'git push origin feat:refs/heads/Main'
run block "$A" 'git push origin feat:refs/heads/MAIN'
run block "$A" 'git push origin feat:refs/heads/backup/y'
run block "$A" 'git push origin feat:refs/tags/v9'
run block "$A" 'git push origin feat:refs/heads/'
run block "$A" 'git push origin :refs/heads/feat'
run block "$A" 'git push origin feat:refs/heads/a:b'
run block "$A" 'git push origin +feat:refs/heads/feat'
run block "$A" 'git push origin feat:refs/heads/main>&1x'
run block "$A" 'git push --force-with-lease origin feat:refs/heads/feat main'
run block "$T/mirrorcfg" 'git push origin feat:refs/heads/feat'
run block "$T/pushurl" 'git push -u origin feat:refs/heads/feat'
run block "$A" 'git push origin main:refs/heads/feat9'  # v1 allow -> v2 block
run allow "$T/mapped" 'git push origin feat:refs/heads/feat'
# earlier wrapper/substitution cases, restated with explicit destinations so
# the explicit-destination rule is not what blocks them
run block "$A" '(git push origin feat:refs/heads/feat)'
run block "$A" "(cd $A && git push origin feat:refs/heads/feat)"
run block "$A" 'x=`git push origin feat:refs/heads/feat`'
run block "$A" 'x=$(git push origin feat:refs/heads/feat)'
run block "$A" 'cat <(git push origin feat:refs/heads/feat)'
run block "$A" 'gh pr checkout 1 && git push origin feat:refs/heads/feat'
run block "$A" 'git checkout main && git push origin HEAD:refs/heads/feat'
run block "$A" 'git status; git switch main; git push origin HEAD:refs/heads/feat'
# case variants are checked exactly like the lowercase command
run block "$A" 'GIT push origin feat:refs/heads/feat'  # v1 allow -> v2 block
run block "$A" 'git PUSH origin feat:refs/heads/feat'  # v1 allow -> v2 block
run block "$A" 'GIT push origin feat:refs/heads/main'
run block "$A" 'git PUSH origin feat:refs/heads/main'
run block "$A" 'echo ok && git push origin feat:refs/heads/main'
run block "$A" 'git push origin feat:refs/heads/feat | cat'  # v1 allow -> v2 block
run block "$A" 'git push origin feat:refs/heads/main | cat'
# --- review round 5 (RT4): companions that write files or set variables,
# submodule recursion on fetch/pull
P='git push origin feat:refs/heads/feat'
run block "$A" "uniq $T/rt4/evil.cfg .git/config; $P"
run block "$A" "sort -o .git/config $T/rt4/evil2.cfg && $P"
run block "$A" "printf -v HOME $T/rt4/fakehome; $P"
run block "$A" "printf -v PATH $T/rt4/fakebin:/usr/bin:/bin && $P"
run block "$T/nestedsub" 'git fetch origin'
run block "$T/nestedsub" 'git pull --no-rebase origin feat-sub'
run block "$T/nestedsub" 'git fetch'
run block "$T/nestedsub" 'git fetch --all'
run block "$T/nestedsub" 'git pull'
run allow "$T/nestedsub" 'git fetch --no-recurse-submodules origin'
run block "$T/nestedsub" 'git pull --no-recurse-submodules --no-rebase origin feat-sub'  # v1 allow -> v2 block
run allow "$T/nestedsub" 'git fetch --recurse-submodules=no --all'
run block "$A" 'git fetch origin'
run block "$A" 'git pull origin feat'
run allow "$A" 'git ls-remote origin'
# same class, found while fixing: redirects and option-driven writes in
# companion segments, git companions with global options or --output, aliases
run block "$A" "echo x >> .git/config; $P"
run block "$A" "cat $T/rt4/evil.cfg > .git/config && $P"
run block "$A" "cat $T/rt4/evil.cfg>.git/config && $P"
run block "$A" "sort --compress-program=$T/rt4/x -S 1 /etc/hosts; $P"
run block "$A" "printf x; $P"
run block "$A" "date; $P"
run block "$A" "tee .git/config; $P"
run block "$A" "git -c core.hooksPath=$T/rt4/hp commit -m x && $P"
run block "$A" "git --exec-path=$T/rt4/fakebin status && $P"
run block "$A" "git log --output=.git/config -1; $P"
run block "$A" "git diff --output=.git/config; $P"
run block "$A" "git show --outp=.git/config; $P"
run block "$A" "git l --output=.git/config; $P"
run block "$A" "git st; $P"
run block "$A" "git commit -m x --template=.git/HEAD && $P"
run block "$A" "git commit -m 'msg' && $P"
run block "$A" "$P # main"
run block "$A" "echo {a,b}; $P"
run block "$A" "$P \; echo"
run block "$A" "echo !!; $P"
run block "$A" '=git push origin main'
run block "$A" 'cd sub && =git push origin feat:refs/heads/main'
run block "$A" "git commit -m wip && $P"
run block "$A" "git add -A && git commit -q -m wip && $P 2>&1 | tail -3"
run block "$A" "git status --short; git rev-parse HEAD; $P"
run allow "$A" "git commit -q -m wip"
run allow "$A" "git status --short"
run block "$A" "echo pushing; $P; echo done"  # v1 allow -> v2 block
run block "$A" "ls; pwd; true; sleep 0; $P | grep -c x; wc -l .git/HEAD; head -1 .git/HEAD; cat .git/HEAD; tail -1 .git/HEAD"  # v1 allow -> v2 block
run allow "$A" "git l --output=$T/rt4/out"
run allow "$A" 'git log --output=/dev/null -1'
# --- review round 6 (RT5-1): git companions in attacker repos reached by cd
run block "$A" "cd $T/rt5/attacker; git commit --allow-empty -m x; cd $A; $P"
run block "$A" "cd $T/rt5/attacker && git commit --allow-empty -m x && cd $A && $P"
run block "$A" "pushd $T/rt5/attacker && git commit --allow-empty -m x && cd $A && $P"
run block "$A" "cd $T/rt5/attacker; git commit --allow-empty -m x; cd $A; git ls-remote origin"
run block "$A" "cd $T/rt5/att2; git status; cd $A; $P"
run block "$A" "cd $T/rt5/att2; git status; cd $A; git fetch --no-recurse-submodules origin"
run block "$A" "cd $T/rt5/att3 && git add f && cd $A && $P"
run block "$A" "git -C $T/rt5/att2 status; $P"
run block "$A" "git -C $T/rt5/attacker rev-parse HEAD && $P"
run allow "$T/rt5/att2" 'git status'   # alone: not a network line, stays sandboxed
# --- v2: exactly one plain command, explicit same-name destinations
run allow "$A" 'git push origin feat:refs/heads/feat'
run allow "$A" 'git  push   origin  feat:refs/heads/feat'
run allow "$A" 'git push -u -v origin feat:refs/heads/feat'
run allow "$A" 'git push --atomic --porcelain origin feat:refs/heads/feat newfeat:refs/heads/newfeat'
run allow "$A/sub" 'git push --set-upstream origin feat:refs/heads/feat'
run allow "$T/allowed-link" 'git push origin feat:refs/heads/feat'
run allow "$A" 'git push -n origin mainx:refs/heads/mainx'
run allow "$A" 'git push origin symb:refs/heads/symb'
run allow "$A" 'git push --force-with-lease=feat2 origin feat2:refs/heads/feat2'
run allow "$T/mapped" 'git push origin feat:refs/heads/feat'
run block "$A" 'git push origin feat:refs/heads/feat2'
run block "$A" 'git push origin feat'
run block "$A" 'git push origin symb'
run block "$A" 'git push origin main:refs/heads/main'
run block "$A" 'git push origin backup/x:refs/heads/backup/x'
run block "$A" 'git push origin nosuch:refs/heads/nosuch'
run block "$A" 'git push origin v1:refs/heads/v1'
run block "$A" 'git push origin HEAD:refs/heads/HEAD'
run block "$A" 'git push origin refs/heads/feat:refs/heads/refs/heads/feat'
run block "$A" 'git push origin -feat:refs/heads/-feat'
run block "$A" 'git push origin +feat:refs/heads/+feat'
run block "$A" 'git push upstream feat:refs/heads/feat'
run block "$A" 'git push --no-verify origin feat:refs/heads/feat'
run block "$A" 'git push -uv origin feat:refs/heads/feat'
run block "$A" 'git push --repo=upstream origin feat:refs/heads/feat'
run block "$A" 'git push -o ci.skip origin feat:refs/heads/feat'
run block "$A" 'git push --force origin feat2:refs/heads/feat2'
run block "$A" 'git push --force-with-lease=feat2:a:b origin feat2:refs/heads/feat2'
run block "$A" 'git push --force-with-lease=-x origin feat2:refs/heads/feat2'
HOOKARGS_SAVE=$HOOKARGS
HOOKARGS=$(printf '%s' "$HOOKARGS" | sed 's/ --allow-force-with-lease//')
run block "$A" 'git push --force-with-lease origin feat2:refs/heads/feat2'
run block "$A" 'git push --force-if-includes origin feat:refs/heads/feat'
run block "$A" 'git push --force-with-lease=feat2 origin feat2:refs/heads/feat2'
HOOKARGS=$HOOKARGS_SAVE
run block "$A" 'git push origin feat2:refs/heads/feat2 feat:refs/heads/feat --mirror'
run block "$A" 'git --no-pager push origin feat:refs/heads/feat'
run block "$A" 'git -C . push origin feat:refs/heads/feat'
run block "$A" 'git -c core.sshCommand=x push origin feat:refs/heads/feat'
run block "$A" 'git push origin feat:refs/heads/feat;'
run block "$A" 'git push origin feat:refs/heads/feat #'
run block "$A" 'git push origin feat:refs/heads/feat &'
run block "$A" "$(printf 'git \\\npush origin main')"
run block "$A" "$(printf 'git push origin feat:refs/heads/feat\n')x"
run block "$A" "$(printf 'git push origin f\303\251at:refs/heads/f\303\251at')"
run block "$A" 'command git push origin feat:refs/heads/feat'
run block "$A" 'exec git push origin feat:refs/heads/feat'
run block "$A" 'FOO=1 git push origin feat:refs/heads/feat'
run block "$A" 'git status && git push origin feat:refs/heads/feat'
run block "$A" 'git archive --remote=origin main'
run block "$A" 'git log --grep push'   # over-block, documented
run allow "$A" 'git log --oneline | grep push'
run allow "$A" 'grep -rn push .git/config'
run allow "$A" "$(printf 'git commit -m x\npush')"
# --- v2: fetch / pull / ls-remote
run allow "$A" 'git fetch --no-recurse-submodules'
run allow "$A" 'git fetch --no-recurse-submodules --prune --tags origin'
run allow "$A" 'git fetch -q --no-recurse-submodules upstream main feat'
run allow "$A" 'git fetch --no-recurse-submodules --dry-run -v origin'
run allow "$A" 'git fetch --recurse-submodules=no --all'
run block "$A" 'git fetch --no-recurse-submodules origin +main'
run block "$A" 'git fetch --no-recurse-submodules origin main:main'
run block "$A" 'git fetch --no-recurse-submodules --all origin'
run block "$A" 'git fetch --no-recurse-submodules origin.git'
run block "$A" 'git fetch --no-recurse-submodules --upload-pack=x origin'
run allow "$T/ffpull" 'git pull --ff-only --no-recurse-submodules origin feat2'
run allow "$A" 'git pull --ff-only --no-recurse-submodules'
run allow "$A" 'git pull -q --ff-only --no-recurse-submodules origin feat'
run block "$A" 'git pull --ff-only origin feat'
run block "$A" 'git pull --ff-only --no-recurse-submodules origin feat feat2'
run block "$A" 'git pull --ff-only --no-recurse-submodules --rebase origin feat'
run block "$A" 'git pull --ff-only --no-recurse-submodules --all'
run block "$A" 'git pull --ff-only --no-recurse-submodules origin feat:feat'
run block "$T/viaurl" 'git pull --ff-only --no-recurse-submodules'
run allow "$A" 'git ls-remote --heads --tags --refs -q --symref origin'
run allow "$A" 'git ls-remote origin main'
run block "$A" 'git ls-remote'
run block "$A" 'git ls-remote --upload-pack=x origin'
# --- v2: repository resolution
printf '%s/attacker/.git\n' "$T" >"$A/.git/worktrees/wt-tamper/commondir"
run block "$T/wt-tamper" 'git push origin wttamper:refs/heads/wttamper'
run block "$T/wt-tamper" 'git fetch --no-recurse-submodules origin'
run block "$T/attacker" 'git push origin feat:refs/heads/feat'
run block "$A/.git/refs" 'git ls-remote origin'
# --- FRT2: a linked worktree whose top level is (inside) a git dir
F=$T/frt2
P='git pull --ff-only --no-recurse-submodules origin evil'
run block "$F/.git" "$P"
run block "$F/.git" 'git fetch --no-recurse-submodules origin'
run block "$F/.git" 'git push origin wbase:refs/heads/wbase'
run block "$F/.git" 'git ls-remote origin'
run block "$F/.git/refs" "$P"
run block "$F/.git/worktrees/frtgd/x" "$P"
run block "$A/.git/frtx" "$P"
run block "$T/outer" 'git pull --ff-only --no-recurse-submodules origin main'
run block "$T/outer" 'git fetch --no-recurse-submodules origin'
run allow "$T/outer/inner" 'git fetch --no-recurse-submodules origin'
run allow "$A/.claude/worktrees/agent1" 'git push origin agentwt:refs/heads/agentwt'
run allow "$A/.claude/worktrees/agent1" 'git fetch --no-recurse-submodules origin'
# live check: if the hook allowed the attack, run it and look for the payload
if [ "$SABOTAGE" = 0 ]; then
  # shellcheck disable=SC2086
  if printf '{"tool_input":{"command":"%s"},"cwd":"%s"}' "$P" "$F/.git" | "$SHP" "$HOOK" $HOOKARGS >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    (cd "$F/.git" && $P >/dev/null 2>&1 </dev/null)
  fi
  if [ -e "$T/frt2-PWNED" ] || [ -e "$F/.git/hooks/post-merge" ]; then
    echo "FAIL FRT2 gitdir worktree: post-merge planted/ran"; fail=$((fail + 1))
  else pass=$((pass + 1)); fi
fi
# session environment that redirects git
export GIT_DIR="$A/.git"
run block "$T/plain" 'git push origin feat:refs/heads/feat'
run block "$A" 'git ls-remote origin'
run allow "$A" 'ls'
unset GIT_DIR
export GIT_CONFIG_PARAMETERS="'remote.origin.pushurl=x'"
run block "$A" 'git push origin feat:refs/heads/feat'
unset GIT_CONFIG_PARAMETERS
run block "$A" 'git push origin feat:refs/heads/feat' --protect
# --- code review (v2): quoted separators between git and its verb
run block "$A" 'git -c "x=)" push origin feat:refs/heads/feat'
run block "$A" "git -c 'x=;' push origin main"
run block "$A" 'git --namespace="|" fetch --no-recurse-submodules origin'
run block "$A" 'git -c "x=&" ls-remote origin'
run block "$A" "$(printf 'git -c "x=\n" push origin main')"
run block "$A" "$(printf "git -c 'a\n' push origin main")"
run block "$A" 'g"i"t -c "x=;" push origin main'
# ANSI-C ($'...') escapes and $"..." spelling git or its verb
run block "$A" "git \$'\\x70ush' upstream main"
run block "$A" "git \$'\\160ush' origin main"
run block "$A" "git \$'\\x70ush' origin \$'\\x3a'feat"
run block "$A" "git \$'\\x70ush' origin --tags"
run block "$A" "git \$'\\x66etch' /tmp/x.git main"
run block "$A" "\$'\\x67it' push upstream main"
run block "$A" "\$'\\147it' push upstream main"
run block "$A" "\$'\\u0067it' push upstream main"
run block "$A" "git \$'push' origin main"
run block "$A" "git \$\"push\" origin main"
run block "$A" "git -c \"x=;\" \$\"push\" origin main"
run block "$A" "git -c \"x=;\" \$'push' origin main"
run allow "$A" "printf '%s' a | tr \$'\\n' ' '"
run allow "$A" "IFS=\$'\\t' read -r a b < /dev/null"
run block "$A" 'FOO="a;b" git push origin main'
run block "$A" 'git -c "alias.zz=push" zz'
run allow "$A" 'git commit -m "a; push"'
run allow "$A" "$(printf 'git commit -m "$(cat <<%sEOF%s\nfix: push the fetch fix\ndo not pull\nEOF\n)"' "'" "'")"
run allow "$A" "$(printf "git commit -F - <<EOF\ndon't push it\nEOF")"
run allow "$A" 'echo "a;b" | grep push'
# --- code review (v2): fail closed on configuration and environment errors
HOOKARGS_SAVE=$HOOKARGS
HOOKARGS=$(printf '%s' "$HOOKARGS" | sed 's/ --protect main --protect backup\/\*//')
run block "$A" 'git push origin feat:refs/heads/feat'
run allow "$A" 'ls'
HOOKARGS=$(printf '%s' "$HOOKARGS_SAVE" | sed 's/--remote origin //')
run block "$A" 'git fetch --no-recurse-submodules origin'
HOOKARGS=$HOOKARGS_SAVE
run block "$T/deleted-dir" 'git push origin feat:refs/heads/feat'
mkdir "$T/gone"; cd "$T/gone" && rmdir "$T/gone"
run allow "$A" 'git push origin feat:refs/heads/feat'   # hook process cwd deleted
run allow "$A" 'ls'
cd "$T" || exit 70
# tools: awk that fails or prints nothing, git missing from PATH
mkdir -p "$T/badawk" "$T/emptyawk" "$T/nogit"
printf '#!/bin/sh\nexit 1\n' >"$T/badawk/awk"
printf '#!/bin/sh\ncat >/dev/null\nexit 0\n' >"$T/emptyawk/awk"
chmod +x "$T/badawk/awk" "$T/emptyawk/awk"
for t in cat awk jq python3 plutil tr sed sleep kill "$SH"; do
  p=$(command -v "$t") && case $p in /*) ln -s "$p" "$T/nogit/$t" ;; esac
done
# (HOOKPATH is the PATH of the hook process only)
HOOKPATH=$T/badawk:$PATH
run block "$A" 'git status'
run allow "$A" 'ls'
HOOKPATH=$T/emptyawk:$PATH
run block "$A" 'git push origin feat:refs/heads/feat'
HOOKPATH=$T/nogit
run block "$A" 'git ls-remote origin'
run allow "$A" 'git status'
HOOKPATH=''
# over-long commands: blocked only when they could mention git
run block "$A" "$(awk 'BEGIN { for (i = 0; i < 4500; i++) print "echo git log of the day" }')"
run allow "$A" "$(awk 'BEGIN { for (i = 0; i < 7000; i++) print "echo hello world" }')"
# a dry run that hangs (ssh never answers) is killed by the watchdog and blocks
printf '#!/bin/sh\nexec sleep 30\n' >"$T/hangssh"; chmod +x "$T/hangssh"
GIT_SSH=$T/hangssh PUSH_GUARD_DRY_RUN_SECS=2; export PUSH_GUARD_DRY_RUN_SECS
t0=$(date +%s)
run block "$T/sshish" 'git push origin feat:refs/heads/feat'
t1=$(date +%s)
GIT_SSH=$T/fakessh; unset PUSH_GUARD_DRY_RUN_SECS
if [ $((t1 - t0)) -gt 10 ]; then
  echo "FAIL watchdog: hook took $((t1 - t0)) s"; fail=$((fail + 1))
fi
# --- summary marker
# --- malformed input / configuration
rawrun block 'invalid JSON mentioning git' '{"tool_input":{"command":"git push origin main"'
rawrun block 'invalid JSON without g-i-t' '{"tool_input":{"command":"ls'  # review: allow -> block (fail closed)
rawrun block 'empty input' ''
rawrun block 'not an object' '[]'
rawrun block 'NUL escape' "{\"tool_input\":{\"command\":\"ls\\u0000\"},\"cwd\":\"$A\"}"
rawrun block 'missing command, mentions git' '{"tool_input":{"cmd":"git push origin main"},"cwd":"/"}'
rawrun block 'unicode-escaped git' "{\"tool_input\":{\"command\":\"\\u0067it push origin main\"},\"cwd\":\"$A\"}"
rawrun allow 'unicode-escaped harmless' "{\"tool_input\":{\"command\":\"\\u006cs -la\"},\"cwd\":\"$A\"}"
rawrun block 'missing cwd' '{"tool_input":{"command":"git push origin feat"}}'
run block "$A" 'git push origin feat:refs/heads/feat' --bogus-flag
run allow "$A" 'ls' --bogus-flag
PUSH_GUARD_JSON=none
run block "$A" 'git push origin feat:refs/heads/feat'
run block "$A" 'git status'
run allow "$A" 'ls -la'
PUSH_GUARD_JSON=$JSONP

# ----------------------------------------------------------- end to end ---
# Re-check each allowed network command with the hook (state changes as the
# pushes land), run it for real, and assert no protected ref moved.
snap() {
  for r in origin upstream; do
    git -C "$R/$r.git" for-each-ref --format="$r %(refname) %(objectname)" \
      refs/heads/main 'refs/heads/backup/*' refs/heads/Main refs/tags
  done
}
ne2e=0 e2efail=0 nrej=0
if [ "$SABOTAGE" = 0 ] && [ -f "$T/e2e.list" ]; then
  snap | grep -q '^origin refs/heads/main ' || { echo 'FAIL e2e snapshot is empty'; fail=$((fail + 1)); }
  while IFS="$(printf '\t')" read -r dir c extra; do
    j=$(printf '{"tool_input":{"command":"%s"},"cwd":"%s"}' "$(json_str "$c")" "$(json_str "$dir")")
    # shellcheck disable=SC2086
    printf '%s' "$j" | "$SH" "$HOOK" $HOOKARGS $extra >/dev/null 2>&1 || continue
    before=$(snap)
    # the command is plain words (the hook checked), so word splitting is exact
    # shellcheck disable=SC2086
    out=$(cd "$dir" && $c 2>&1 </dev/null); erc=$?
    after=$(snap)
    ne2e=$((ne2e + 1))
    if [ "$before" != "$after" ]; then
      e2efail=$((e2efail + 1)); printf 'FAIL e2e protected ref moved: (%s) %s\n' "$dir" "$c"
    elif [ "$erc" != 0 ] && printf '%s' "$out" | grep -q '\[rejected\]'; then
      nrej=$((nrej + 1))   # a doomed non-fast-forward push is harmless
    elif [ "$erc" != 0 ]; then
      e2efail=$((e2efail + 1)); printf 'FAIL e2e command failed rc=%s: (%s) %s\n      %s\n' "$erc" "$dir" "$c" "$(printf '%s' "$out" | tail -1)"
    fi
  done <"$T/e2e.list"
  fail=$((fail + e2efail))
fi

# --------------------------------------------------------------- summary ---
if [ -e "$T/rt5/PWNED" ]; then
  echo "FAIL the hook itself ran an attacker repo hook/fsmonitor/filter ($T/rt5/PWNED)"
  fail=$((fail + 1))
fi
if [ "$SABOTAGE" = 1 ]; then
  printf 'sabotage: %s of %s block cases failed against an always-allow stub\n' "$blockfail" "$nblock"
  [ "$blockfail" = "$nblock" ] && [ "$nblock" -gt 0 ] && { rm -rf "$WORK"; exit 0; }
  exit 1
fi
printf '%s: shell=%s json=%s pass=%s fail=%s (block cases=%s, e2e runs=%s rejected-nonff=%s e2e fail=%s)\n' "${0##*/}" "$SH" "$JSONP" "$pass" "$fail" "$nblock" "$ne2e" "$nrej" "$e2efail"
[ "$fail" = 0 ] && rm -rf "$WORK"
[ "$fail" = 0 ]
