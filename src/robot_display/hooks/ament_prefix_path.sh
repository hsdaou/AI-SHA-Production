# generated from colcon_core/shell/template/hook_env.sh.em

# prepend the install prefix to AMENT_PREFIX_PATH
COLCON_CURRENT_PREFIX="${COLCON_CURRENT_PREFIX:-$(dirname "$(dirname "$(dirname "$0")")")}"
_colcon_prepend_unique_value AMENT_PREFIX_PATH "$COLCON_CURRENT_PREFIX"
