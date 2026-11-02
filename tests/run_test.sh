#!/usr/bin/env bash
# 真机数值验证：编译并运行 AddCustom（z = x + y），Ascend 910B3
set -o pipefail

source /home/developer/Ascend/ascend-toolkit/set_env.sh || { echo "set_env.sh 失败"; exit 1; }
CUST="$HOME/Ascend/custom_opp/vendors/customize"
SRC_DIR="/mnt/workspace/CANNTEST/tests"
cd "$SRC_DIR" || exit 1

echo "=== 环境 ==="
echo "  ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "  CUST=$CUST"
ls "$ASCEND_HOME_PATH/include/acl/acl.h" >/dev/null 2>&1 && echo "  acl.h OK" || echo "  acl.h 缺失"
ls "$CUST/op_api/include/aclnn_add_custom.h" >/dev/null 2>&1 && echo "  aclnn_add_custom.h OK" || echo "  自定义头缺失"

echo "=== 编译 ==="
g++ -std=c++17 -O2 test_add_custom.cpp -o test_add_custom \
  -I"$ASCEND_HOME_PATH/include" \
  -I"$CUST/op_api/include" \
  -L"$CUST/op_api/lib" -lcust_opapi \
  -L"$ASCEND_HOME_PATH/lib64" -lascendcl -lnnopbase -lpthread 2>&1 | head -20
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ] || [ ! -x ./test_add_custom ]; then
  echo "  编译失败 rc=$rc"
  exit 1
fi
echo "  编译成功: $(ls -lh test_add_custom | awk '{print $5}')"

echo "=== 运行（真机） ==="
if [ -f "$CUST/bin/set_env.bash" ]; then
  # shellcheck disable=SC1090
  source "$CUST/bin/set_env.bash" >/dev/null 2>&1
fi
export ASCEND_CUSTOM_OPP_PATH="$CUST"
export LD_LIBRARY_PATH="$CUST/op_api/lib:$CUST/op_proto/lib:$CUST/op_proto/lib/linux/aarch64:$LD_LIBRARY_PATH"
timeout 240 ./test_add_custom
echo "段错误==退出码139 / 正常==0"
echo "run_rc=$?"
