#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <cuda.h>
#include <unistd.h>

#include <future>
#include <iostream>
#include <map>
#include <mutex>

using namespace tensorflow;

REGISTER_OP("GetTensorAddr")
    .Input("tensor: T")
    .Output("addr: int64")
    .Attr("T: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        c->set_output(0, c->Vector(2));
        return Status::OK();
    });

class GetTensorAddrOp : public OpKernel {
public:
    explicit GetTensorAddrOp(OpKernelConstruction* context)
    : OpKernel(context) 
    {}

    void Compute(OpKernelContext* context) override 
    {
        const Tensor& input_tensor = context->input(0);
        StringPiece input_tensor_data = input_tensor.tensor_data();
        CUdeviceptr device_ptr = reinterpret_cast<CUdeviceptr>(input_tensor_data.data());
        size_t size = input_tensor_data.size();

        Tensor *addr_tensor = nullptr;
        TensorShape shape = TensorShape();
        shape.AddDim(2);
        context->allocate_output(0, shape, &addr_tensor);
        addr_tensor->vec<int64>()(0) = static_cast<int64>(device_ptr);
        addr_tensor->vec<int64>()(1) = static_cast<int64>(size);
    }
};

REGISTER_KERNEL_BUILDER(Name("GetTensorAddr").Device(DEVICE_GPU).HostMemory("addr"), GetTensorAddrOp);

class StreamSingleton {
private:
    static StreamSingleton* instance;
    static std::once_flag init_instance_flag;

    std::map<std::tuple<bool, CUdevice>, CUstream *> stream_list;
    std::map<std::tuple<bool, CUdevice>, std::mutex *> stream_mutex_list;
    std::mutex list_mutex;

    StreamSingleton()
    {}

public:
    static StreamSingleton& getInstance()
    {
        std::call_once(
            init_instance_flag, 
            [&]() -> void {
                instance = new StreamSingleton;
            }
        );
        return *instance;
    }

    void claim(bool is_prefetch, CUdevice device) 
    {
        std::mutex *stream_mutex_ptr;
        std::tuple<bool, CUdevice> tup(is_prefetch, device);
        list_mutex.lock();
        if (stream_mutex_list.count(tup) == 0) {
            stream_mutex_list[tup] = new std::mutex;
        }
        stream_mutex_ptr = stream_mutex_list[tup];
        list_mutex.unlock();
        
        stream_mutex_ptr->lock();
    }

    void release(bool is_prefetch, CUdevice device)
    {
        std::tuple<bool, CUdevice> tup(is_prefetch, device);
        list_mutex.lock();
        stream_mutex_list[tup]->unlock();
        list_mutex.unlock();
    }

    CUstream& getStream(bool is_prefetch, CUdevice device)
    {
        CUstream *stream_ptr;
        std::tuple<bool, CUdevice> tup(is_prefetch, device);
        list_mutex.lock();
        if (stream_list.count(tup) == 0) {
            stream_list[tup] = new CUstream;
            cuStreamCreate(stream_list[tup], CU_STREAM_NON_BLOCKING);
        }
        stream_ptr = stream_list[tup];
        list_mutex.unlock();
        return *stream_ptr;
    }
};

StreamSingleton* StreamSingleton::instance = nullptr;
std::once_flag StreamSingleton::init_instance_flag;

struct MemRange {
    static const int64 PAGE_SIZE;
    int64 addr;
    int64 size;

    MemRange(int64 addr, int64 size)
    : addr(addr), size(size)
    {}

    bool operator<(const MemRange& rhs) const
    {
        return this->addr < rhs.addr;
    }

    bool isOverlapped(const MemRange& rhs) const
    {
        return !(rhs.addr > this->addr + this->size) && !(this->addr > rhs.addr + rhs.size);
    }

    void mergeWith(const MemRange& rhs)
    {
        int64 new_addr = std::min(this->addr, rhs.addr);
        int64 new_addr_end = std::max(this->addr + this->size, rhs.addr + rhs.size);
        int64 new_size = new_addr_end - new_addr;

        this->addr = new_addr;
        this->size = new_size;
    }

    MemRange &toPageAligned(bool expand_or_shrink)
    {
        if (expand_or_shrink) {
            int64 begin_addr = addr - addr % PAGE_SIZE;
            int64 end_addr = addr + size; 
            end_addr = end_addr - end_addr % PAGE_SIZE + ((end_addr % PAGE_SIZE == 0) ? 0 : PAGE_SIZE);

            addr = begin_addr;
            size = end_addr - begin_addr;
        }
        else {
            int64 addr_offset = addr % PAGE_SIZE;
            int64 begin_addr = (addr_offset == 0 ? addr : addr - addr_offset + PAGE_SIZE);
            int64 end_addr = addr + size;
            int64 end_addr_offset = end_addr % PAGE_SIZE;
            end_addr = (end_addr_offset == 0 ? end_addr : end_addr - end_addr_offset);

            addr = begin_addr;
            size = end_addr - begin_addr;
        }

        return *this;
    }
};

//const int64 MemRange::PAGE_SIZE = sysconf(_SC_PAGESIZE); 
const int64 MemRange::PAGE_SIZE = 2LL * 1024LL * 1024LL;

REGISTER_OP("Prefetch")
    .Input("addrs: int64")
    .Attr("expand_or_shrink: bool")
    .SetShapeFn(shape_inference::NoOutputs);

class PrefetchOp : public AsyncOpKernel {
private:
    bool expand_or_shrink;

public:
    explicit PrefetchOp(OpKernelConstruction* context)
    : AsyncOpKernel(context)
    {
        context->GetAttr("expand_or_shrink", &expand_or_shrink);
    }

    void ComputeAsync(OpKernelContext* context, DoneCallback done) override
    {
        const Tensor& addrs = context->input(0);

        // Merge intervals of requested addresses
        int64 num_range = addrs.dim_size(0);
        auto addrs_tensor = addrs.tensor<int64, 2>();

        const int TAG_BEGIN = 0;
        const int TAG_END = 1;
        
        struct AddrWithTag {
            int64 addr;
            int tag;

            AddrWithTag(int64 addr, int tag)
            : addr(addr), tag(tag)
            {}

            bool operator<(const AddrWithTag &rhs) const
            {
                return this->addr < rhs.addr;
            }
        };

        std::vector<AddrWithTag> addr_list;
        for (int64 i = 0; i < num_range; i++) {
            if (addrs_tensor(i, 1) <= 0)
                continue;

            MemRange mem_range(addrs_tensor(i, 0), addrs_tensor(i, 1));
            mem_range.toPageAligned(expand_or_shrink); 

            addr_list.emplace_back(mem_range.addr, TAG_BEGIN);
            addr_list.emplace_back(mem_range.addr + mem_range.size, TAG_END);
        }
        std::sort(addr_list.begin(), addr_list.end());

        std::vector<MemRange> range_list;
        int depth = 0;
        int64 addr_begin = 0LL;
        for (AddrWithTag &addr_with_tag : addr_list) {
            int64 addr = addr_with_tag.addr;
            int tag = addr_with_tag.tag;

            switch (tag) {
            case TAG_BEGIN:
                depth += 1;
                if (depth == 1) {
                    addr_begin = addr;
                }
                break;

            case TAG_END:
                depth -= 1;
                if (depth == 0) {
                    range_list.emplace_back(addr_begin, addr - addr_begin);
                }
                break;

            default:
                assert(false);
            }
        }

        // Issue prefetches
        const tensorflow::DeviceBase::GpuDeviceInfo *device_info =
            context->device()->tensorflow_gpu_device_info();
        int device_ordinal = device_info->gpu_id;
        CUdevice cu_device = -1;
        cuDeviceGet(&cu_device, device_ordinal);

        CUevent event;
        cuEventCreate(&event, CU_EVENT_DEFAULT);
        StreamSingleton &stream_singleton = StreamSingleton::getInstance();
        stream_singleton.claim(true, cu_device);
        CUstream &cu_stream = stream_singleton.getStream(true, cu_device);

        for (MemRange& range : range_list) {
            //std::cerr << "Prefetch: " << range.addr << ", " << range.size << std::endl;
            cuMemPrefetchAsync(range.addr, range.size, cu_device, cu_stream);
        }
        cuEventRecord(event, cu_stream);
        stream_singleton.release(true, cu_device);
        cuEventSynchronize(event);
        cuEventDestroy(event);
        //std::cerr << "---" << std::endl;

        done();
    }
};

REGISTER_KERNEL_BUILDER(Name("Prefetch").Device(DEVICE_GPU).HostMemory("addrs"), PrefetchOp);

REGISTER_OP("Evict")
    .Input("addrs: int64")
    .Input("exclude_addrs: int64")
    .SetShapeFn(shape_inference::NoOutputs);

class EvictOp : public AsyncOpKernel {
public:
    explicit EvictOp(OpKernelConstruction* context)
    : AsyncOpKernel(context)
    {}

    void ComputeAsync(OpKernelContext* context, DoneCallback done) override
    {
        const Tensor& addrs = context->input(0);
        const Tensor& exclude_addrs = context->input(1);

        // Extract addresses
        const int TAG_EVICT_BEGIN = 0;
        const int TAG_EVICT_END = 1;
        const int TAG_EXCLUDE_BEGIN = 2;
        const int TAG_EXCLUDE_END = 3;

        int64 num_addrs = addrs.dim_size(0);
        auto addrs_tensor = addrs.tensor<int64, 2>();
        int64 num_exclude_addrs = exclude_addrs.dim_size(0);
        auto exclude_addrs_tensor = exclude_addrs.tensor<int64, 2>();

        struct AddrWithTag {
            int64 addr;
            int tag;

            AddrWithTag(int64 addr, int tag)
            : addr(addr), tag(tag)
            {}

            bool operator<(const AddrWithTag &rhs) const
            {
                return this->addr < rhs.addr;
            }
        };

        std::vector<AddrWithTag> addr_list;
        for (int64 i = 0; i < num_addrs; i++) {
            if (addrs_tensor(i, 1) <= 0)
                // Remove size <= 0 requests
                continue;

            MemRange mem_range(addrs_tensor(i, 0), addrs_tensor(i, 1));
            mem_range.toPageAligned(false);

            AddrWithTag addr_begin(mem_range.addr, TAG_EVICT_BEGIN);
            addr_list.push_back(addr_begin);
            AddrWithTag addr_end(mem_range.addr + mem_range.size, TAG_EVICT_END);
            addr_list.push_back(addr_end);
        }
        for (int64 i = 0; i < num_exclude_addrs; i++) {
            if (exclude_addrs_tensor(i, 1) <= 0)
                // Remove size <= 0 requests
                continue;

            MemRange mem_range(exclude_addrs_tensor(i, 0), exclude_addrs_tensor(i, 1));
            mem_range.toPageAligned(false);

            AddrWithTag addr_begin(mem_range.addr, TAG_EXCLUDE_BEGIN);
            addr_list.push_back(addr_begin);
            AddrWithTag addr_end(mem_range.addr + mem_range.size, TAG_EXCLUDE_END);
            addr_list.push_back(addr_end);
        }

        std::sort(addr_list.begin(), addr_list.end());

        // Deal with exclusions
        std::vector<MemRange> range_list;
        int evict_depth = 0;
        int exclude_depth = 0;
        int64 evict_begin = 0LL;
        for (int64 i = 0; i < addr_list.size(); i++) {
            AddrWithTag &addr_with_tag = addr_list[i];
            int tag = addr_with_tag.tag;
            int64 addr = addr_with_tag.addr;
            
            switch (tag) {
            case TAG_EVICT_BEGIN:
                evict_depth += 1;
                if (evict_depth == 1 && exclude_depth == 0)
                    evict_begin = addr;
                break;

            case TAG_EVICT_END:
                evict_depth -= 1;
                if (evict_depth == 0 && exclude_depth == 0 && addr - evict_begin > 0)
                    range_list.emplace_back(evict_begin, addr - evict_begin);
                break;

            case TAG_EXCLUDE_BEGIN:
                exclude_depth += 1;
                if (exclude_depth == 1 && evict_depth > 0 && addr - evict_begin > 0)
                    range_list.emplace_back(evict_begin, addr - evict_begin);
                break;

            case TAG_EXCLUDE_END:
                exclude_depth -= 1;
                if (exclude_depth == 0 && evict_depth > 0)
                    evict_begin = addr;
                break;

            default:
                assert(false);
            }
        }

        // Issue evictions
        const tensorflow::DeviceBase::GpuDeviceInfo *device_info =
            context->device()->tensorflow_gpu_device_info();
        int device_ordinal = device_info->gpu_id;
        CUdevice cu_device = -1;
        cuDeviceGet(&cu_device, device_ordinal);

        CUevent event;
        cuEventCreate(&event, CU_EVENT_DEFAULT);
        StreamSingleton &stream_singleton = StreamSingleton::getInstance();
        stream_singleton.claim(false, cu_device);
        CUstream &cu_stream = stream_singleton.getStream(false, cu_device);

        for (MemRange& range : range_list) {
            CUresult res = cuMemPrefetchAsync(range.addr, range.size, CU_DEVICE_CPU, cu_stream);
            //std::cerr << res << " " << range.addr << " " << range.size << std::endl;
        }
        cuEventRecord(event, cu_stream);
        stream_singleton.release(false, cu_device);
        cuEventSynchronize(event);
        cuEventDestroy(event);

        done();
    }
};

REGISTER_KERNEL_BUILDER(Name("Evict").Device(DEVICE_GPU).HostMemory("addrs").HostMemory("exclude_addrs"), EvictOp);
