
#include "../op_kernel/add_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  AddCustomTilingData *tiling = context->GetTilingData<AddCustomTilingData>();
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int64_t data_sz = 1;
  for (size_t i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);

  // 单次搬运字节数（按 dtype 换算成元素数，避免 fp16 只搬一半字节）
  constexpr uint32_t TILE_BYTES = 16384;
  uint32_t total = static_cast<uint32_t>(data_sz);
  ge::DataType dt = context->GetInputDesc(0)->GetDataType();
  uint32_t dsize = (dt == ge::DT_FLOAT16) ? 2u : 4u;
  uint32_t tileLength = TILE_BYTES / dsize;
  // 小 shape 用单核，避免 blockDim 过大导致每核分不到数据
  uint32_t blockDim = (total < tileLength) ? 1u : 8u;

  tiling->totalLength = total;
  tiling->tileLength = tileLength;
  tiling->tileNum = (total + tileLength - 1u) / tileLength;

  context->SetBlockDim(blockDim);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(AddCustom);
}
