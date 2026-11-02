/* -------------------------------------------------------------------------
 * Ascend C kernel: z = x + y
 * 生成骨架由 msopgen 提供，此处为实际实现：
 *  - 按 block 切分，块内按 tileLength 分块，双缓冲搬运
 *  - 尾部用 DataCopyPad 处理非 32B 对齐长度
 * ------------------------------------------------------------------------- */
#include "kernel_operator.h"
#include "add_custom_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddCustomTilingData* t)
    {
        uint32_t total = t->totalLength;
        uint32_t blockNum = GetBlockNum();
        uint32_t idx = GetBlockIdx();
        // 每核长度按 32B（8 个 fp32 元素）向上对齐，保证 GM 起始地址对齐
        uint32_t per = (total + blockNum - 1) / blockNum;
        per = (per + 7u) / 8u * 8u;
        uint32_t offset = idx * per;
        len_ = (offset >= total) ? 0u : ((total - offset) < per ? (total - offset) : per);
        tileLen_ = t->tileLength;

        xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)x + offset, len_);
        yGm_.SetGlobalBuffer((__gm__ DTYPE_Y*)y + offset, len_);
        zGm_.SetGlobalBuffer((__gm__ DTYPE_Z*)z + offset, len_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLen_ * sizeof(DTYPE_X));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLen_ * sizeof(DTYPE_Y));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLen_ * sizeof(DTYPE_Z));
    }

    __aicore__ inline void Process()
    {
        if (len_ == 0u) {
            return;
        }
        uint32_t loop = (len_ + tileLen_ - 1u) / tileLen_;
        for (uint32_t i = 0; i < loop; ++i) {
            uint32_t cur = (i == loop - 1u) ? (len_ - i * tileLen_) : tileLen_;
            CopyIn(i, cur);
            Compute(cur);
            CopyOut(i, cur);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t i, uint32_t cur)
    {
        LocalTensor<DTYPE_X> lx = inQueueX_.AllocTensor<DTYPE_X>();
        LocalTensor<DTYPE_Y> ly = inQueueY_.AllocTensor<DTYPE_Y>();
        // 非 32B 整数倍的长度用 DataCopyPad，避免越界/丢数
        DataCopyExtParams p{1, static_cast<uint32_t>(cur * sizeof(DTYPE_X)), 0, 0, 0};
        DataCopyPadExtParams<DTYPE_X> padX{false, 0, 0, 0};
        DataCopyPadExtParams<DTYPE_Y> padY{false, 0, 0, 0};
        DataCopyPad(lx, xGm_[i * tileLen_], p, padX);
        DataCopyPad(ly, yGm_[i * tileLen_], p, padY);
        inQueueX_.EnQue(lx);
        inQueueY_.EnQue(ly);
    }

    __aicore__ inline void Compute(uint32_t cur)
    {
        LocalTensor<DTYPE_X> lx = inQueueX_.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> ly = inQueueY_.DeQue<DTYPE_Y>();
        LocalTensor<DTYPE_Z> lz = outQueueZ_.AllocTensor<DTYPE_Z>();
        Add(lz, lx, ly, static_cast<int32_t>(cur));
        outQueueZ_.EnQue<DTYPE_Z>(lz);
        inQueueX_.FreeTensor(lx);
        inQueueY_.FreeTensor(ly);
    }

    __aicore__ inline void CopyOut(uint32_t i, uint32_t cur)
    {
        LocalTensor<DTYPE_Z> lz = outQueueZ_.DeQue<DTYPE_Z>();
        DataCopyExtParams p{1, static_cast<uint32_t>(cur * sizeof(DTYPE_Z)), 0, 0, 0};
        DataCopyPad(zGm_[i * tileLen_], lz, p);
        outQueueZ_.FreeTensor(lz);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;
    GlobalTensor<DTYPE_X> xGm_;
    GlobalTensor<DTYPE_Y> yGm_;
    GlobalTensor<DTYPE_Z> zGm_;
    uint32_t len_ = 0;
    uint32_t tileLen_ = 0;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelAddCustom op;
    op.Init(x, y, z, &tilingData);
    op.Process();
}
