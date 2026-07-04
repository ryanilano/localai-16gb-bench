# ===========================================================================
# ini.sh — minimal INI parser for models.ini. Pure bash + awk, no deps.
# Sourced by prefetch.sh, run-bench.sh and run-quality.sh.
#
# API:
#   ini_sections                 -> model labels, one per line ([*] excluded)
#   ini_get <section> <key>      -> value; falls back to [*]; empty if unset
#   ini_flags <section> <consumer>
#                                -> pass-through "--key value" pairs, one pair
#                                   per line ("--key" then "value"), for the
#                                   given consumer: bench | quality.
#                                   Reads the flags into an array with
#                                   mapfile -t FLAGS < <(ini_flags ...) and
#                                   expand as "${FLAGS[@]}" — never word-split.
#
# Rules encoded here (keep in sync with the header of models.ini):
#   - [*] defaults, per-section override wins
#   - reserved keys (hf,type,sys,template) never emitted by ini_flags
#   - unprefixed keys -> both consumers; bench./quality. -> that consumer only
#   - empty value skips the key; 0 does NOT skip
#   - n-cpu-moe emitted only when type=moe
#   - full-line ;/# comments; inline comments not supported
# ===========================================================================

INI_FILE="${INI_FILE:-./models.ini}"

# llama.cpp's arg parsers (llama-bench's own, and common/arg.cpp for the server)
# match registered option strings EXACTLY: -ngl is valid, --ngl is not;
# --n-cpu-moe is valid, -n-cpu-moe is not. So keys listed in SHORT_FLAGS are
# emitted with one dash, everything else with two. Extend the list if you add
# a short-form llama.cpp option to models.ini.
SHORT_FLAGS="${SHORT_FLAGS:-c t b n p d r s fa ub np ngl ctk ctv}"

_ini_awk() {  # $1=mode $2=section $3=key-or-consumer
  awk -v mode="$1" -v want_sec="$2" -v want_key="$3" -v short_flags="$SHORT_FLAGS" '
    function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
    /^[ \t]*[;#]/ { next }                          # full-line comment
    /^[ \t]*\[/ {
      sec = trim($0); sub(/^\[/, "", sec); sub(/\][ \t]*$/, "", sec)
      if (mode == "sections" && sec != "*") print sec
      next
    }
    /=/ {
      eq = index($0, "=")
      key = trim(substr($0, 1, eq - 1))
      val = trim(substr($0, eq + 1))
      if (key == "") next
      if (mode == "get") {
        if (sec == "*"      && key == want_key) def = val
        if (sec == want_sec && key == want_key) { ovr = val; has_ovr = 1 }
      }
      if (mode == "flags" && (sec == "*" || sec == want_sec)) {
        # scope: strip a matching consumer prefix; drop the other consumer''s keys
        scope = "both"
        if (key ~ /^bench\./)   { scope = "bench";   sub(/^bench\./,   "", key) }
        if (key ~ /^quality\./) { scope = "quality"; sub(/^quality\./, "", key) }
        if (scope != "both" && scope != want_key) next
        if (key == "hf" || key == "type" || key == "sys" || key == "template") {
          if (key == "type" && sec == want_sec) sec_type = val
          if (key == "type" && sec == "*" && !sec_type_set) { def_type = val }
          next
        }
        # record with precedence: section value overrides [*] value
        if (sec == "*") { if (!(key in ovr_v)) def_v[key] = val; def_seen[key] = 1; def_val[key] = val }
        else            { ovr_v[key] = val; ovr_seen[key] = 1 }
        order[++n] = key   # remember first-seen order (dupes deduped on output)
      }
    }
    END {
      if (mode == "get") { print (has_ovr ? ovr : def); exit }
      if (mode == "flags") {
        # resolve effective type for the MoE gate
        # (type is re-read via a second pass in shell; see ini_flags below)
        nshort = split(short_flags, sf, " ")
        for (j = 1; j <= nshort; j++) is_short[sf[j]] = 1
        for (i = 1; i <= n; i++) {
          k = order[i]
          if (k in done) continue; done[k] = 1
          v = (k in ovr_seen) ? ovr_v[k] : def_val[k]
          if (v == "") continue                     # empty value skips; 0 does not
          print ((k in is_short) ? "-" : "--") k; print v
        }
      }
    }
  ' "$INI_FILE"
}

ini_sections() { _ini_awk sections "" ""; }

ini_get() { _ini_awk get "$1" "$2"; }

ini_flags() {  # $1=section $2=bench|quality
  local sec="$1" consumer="$2" type
  type="$(ini_get "$sec" type)"
  _ini_awk flags "$sec" "$consumer" | awk -v type="$type" '
    # gate n-cpu-moe on type=moe: drop the flag AND its value line for dense
    $0 == "--n-cpu-moe" { if (type != "moe") { skip = 1; next } }
    skip { skip = 0; next }
    { print }
  '
}
