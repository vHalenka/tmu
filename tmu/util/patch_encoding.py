"""Shared patch sizes and packed literals for CPU and GPU classifiers.

TMU uses legacy width-first dimensions; keep the same flat pixel addressing.
CPU models call encode_cpu. GPU backends call stage_patches directly.
"""
from numbers import Integral
import numpy as np


def is_multiscale(value):
    return isinstance(value,(tuple,list)) and bool(value) and isinstance(value[0],(tuple,list))


def normalize_patch_dim(value):
    if value is None:return None
    if not isinstance(value,(tuple,list)) or not value:
        raise ValueError('patch_dim must be a pair or a nonempty list of pairs')
    sizes=value if is_multiscale(value) else [value]
    result=[]
    for size in sizes:
        if (not isinstance(size,(tuple,list)) or len(size)!=2 or
            any(not isinstance(v,Integral) or isinstance(v,(bool,np.bool_)) or v<1 for v in size)):
            raise ValueError('Each patch size must contain two positive integers')
        result.append(tuple(map(int,size)))
    if len(set(result))!=len(result):raise ValueError('Duplicate patch sizes are not allowed')
    return result[0] if len(result)==1 else tuple(result)


def normalize_invariance(value):
    if not isinstance(value, str) or value not in ('none', 'horizontal', 'vertical', 'both'):
        raise ValueError('Reflection invariance must be none, horizontal, vertical or both')
    return value


def reflection_axes(value):
    """Bit 0 reverses columns; bit 1 reverses rows."""
    return {'none': 0, 'horizontal': 1, 'vertical': 2, 'both': 3}[normalize_invariance(value)]


def _canonical(values, mode, spatial_axis):
    """Select one intact lexicographic minimum over the full reflection group.

    Every orientation is taken from the original, never a partially chosen
    intermediate. This matters for the four-element horizontal/vertical group.
    """
    axes = reflection_axes(mode)
    if not axes:
        return values
    prefix = values.shape[:spatial_axis]
    result = values
    for orientation in range(1, 4):
        if orientation & ~axes:
            continue
        index = [slice(None)] * values.ndim
        if orientation & 1:
            index[spatial_axis + 1] = slice(None, None, -1)
        if orientation & 2:
            index[spatial_axis] = slice(None, None, -1)
        candidate = values[tuple(index)]
        a = (result == 1).reshape(*prefix, -1)
        b = (candidate == 1).reshape(*prefix, -1)
        first = np.argmax(a != b, axis=-1)[..., None]
        choose = np.take_along_axis(b, first, axis=-1) < np.take_along_axis(a, first, axis=-1)
        choose = choose.reshape(*prefix, *((1,) * (values.ndim - spatial_axis)))
        result = np.where(choose, candidate, result)
    return result


def _reflection_property(name):
    def get(self):
        return self.__dict__.get(name, 'none')
    def set(self, value):
        value = normalize_invariance(value)
        allocated = getattr(self, 'initialized', False) or 'number_of_ta_chunks' in self.__dict__
        if allocated and value != get(self):
            raise ValueError('Reflection invariance is immutable after initialization; create a new model')
        self.__dict__[name] = value
    return property(get, set)


class PatchDimensions:
    image_invariance = _reflection_property('image_invariance')
    patch_invariance = _reflection_property('patch_invariance')

    @property
    def patch_dim(self):return self.__dict__.get('patch_dim')

    @patch_dim.setter
    def patch_dim(self,value):
        value=normalize_patch_dim(value)
        current=self.__dict__.get('patch_dim')
        allocated=getattr(self,'initialized',False) or 'number_of_ta_chunks' in self.__dict__
        if allocated and value!=current and (is_multiscale(current) or is_multiscale(value)):
            raise ValueError('Multiple patch sizes are immutable after initialization; create a new model')
        self.__dict__['patch_dim']=value


def layout(dim,patch_dim):
    w,h,depth=map(int,dim)
    patch_dim=normalize_patch_dim(patch_dim)
    sizes=patch_dim if is_multiscale(patch_dim) else (patch_dim,)
    if min(w,h,depth)<1 or any(pw>w or ph>h for pw,ph in sizes):
        raise ValueError('Patch sizes must fit the spatial input dimensions')
    counts=tuple((w-pw+1)*(h-ph+1) for pw,ph in sizes)
    mw,mh=max(p[0] for p in sizes),max(p[1] for p in sizes)
    nx,ny=max(w-pw for pw,ph in sizes),max(h-ph for pw,ph in sizes)
    features=nx+ny+mw*mh*depth
    if len(sizes)>1:features+=mw*mh+len(sizes)
    return sizes,counts,mw,mh,nx,ny,features


def geometry(dim,patch_dim):
    sizes,counts,mw,mh,nx,ny,features=layout(dim,patch_dim)
    return features,sum(counts)


def _raw(X,dim):
    raw=np.asarray(X)
    flat = raw.ndim == 2 and tuple(dim) == (raw.shape[1],1,1)
    if not flat and (raw.ndim not in (3,4) or tuple(raw.shape[1:3])!=tuple(dim[:2]) or
        raw.size!=len(raw)*int(np.prod(dim))):
        raise ValueError('Patch encoding requires matching spatial image input')
    return np.ascontiguousarray(raw,dtype=np.uint32)


def encode_cpu(X,dim,patch_dim,append_negated=True,image_invariance="none",patch_invariance="none"):
    """Pack all scales on the CPU; no GPU import or device work occurs here."""
    image_invariance = normalize_invariance(image_invariance)
    patch_invariance = normalize_invariance(patch_invariance)
    sizes,counts,mw,mh,nx,ny,features=layout(dim,patch_dim)
    if len(sizes)==1 and image_invariance==patch_invariance=="none":
        from tmu.tools import encode
        chunks=(features*(2 if append_negated else 1)+31)//32
        return encode(X,len(X),sum(counts),chunks,dim,sizes[0],0,append_negated)
    raw=_raw(X,dim)
    if image_invariance != 'none' and len(raw):
        # Whole images follow caller NHW/NHWC rows (axis 1), columns (axis 2).
        raw = _canonical(raw, image_invariance, 1)
    w,h,depth=map(int,dim)
    patches=sum(counts)
    chunks=(features*(2 if append_negated else 1)+31)//32
    result=np.zeros((len(raw),patches,chunks),np.uint32)
    batch_rows=max(1,(32*1024**2)//max(max(counts)*features*3,1))
    offset=0
    for scale,((pw,ph),count) in enumerate(zip(sizes,counts)):
        for start in range(0,len(raw),batch_rows):
            image=raw[start:start+batch_rows].reshape(-1,h,w,depth)==1
            windows=np.lib.stride_tricks.sliding_window_view(image,(ph,pw),axis=(1,2))
            windows=windows.transpose(0,1,2,4,5,3).reshape(len(image),count,ph,pw,depth)
            if patch_invariance != 'none':
                windows = _canonical(windows, patch_invariance, 2)
            positive=np.zeros((len(image),count,features),np.uint8)
            ids=np.arange(count)
            y,x=ids//(w-pw+1),ids%(w-pw+1)
            positive[:,:,:ny]=y[None,:,None]>np.arange(ny)
            positive[:,:,ny:ny+nx]=x[None,:,None]>np.arange(nx)
            content=positive[:,:,ny+nx:ny+nx+mw*mh*depth].reshape(len(image),count,mh,mw,depth)
            content[:,:,:ph,:pw]=windows
            if len(sizes)>1:
                valid=positive[:,:,ny+nx+mw*mh*depth:-len(sizes)].reshape(len(image),count,mh,mw)
                valid[:,:,:ph,:pw]=1
                positive[:,:,-len(sizes)+scale]=1
            bits=np.zeros((len(image),count,chunks*32),np.uint8)
            bits[:,:,:features]=positive
            if append_negated:bits[:,:,features:2*features]=1-positive
            result[start:start+len(image),offset:offset+count]=np.packbits(bits,axis=2,bitorder='little').view('<u4')
        offset+=count
    return result.reshape(len(raw),patches*chunks)


def stage_patches(X,dim,patch_dim,driver,image_invariance="none",patch_invariance="none"):
    """Upload pixels and pack their patch predicates on the GPU."""
    from tmu.clause_bank.clause_bank_cuda import load_cuda_kernel
    image_axes, patch_axes = reflection_axes(image_invariance), reflection_axes(patch_invariance)
    sizes,counts,mw,mh,nx,ny,features=layout(dim,patch_dim)
    raw=_raw(X,dim)
    w,h,depth=map(int,dim)
    patches=sum(counts);chunks=(features*2+31)//32
    output=driver.mem_alloc(max(4,len(raw)*patches*chunks*4))
    if not len(raw):return output
    module=load_cuda_kernel([],'cuda/patch_encoding.cu')
    pack=module.get_function('tc_encode_patch_scale');pack.prepare('PPP'+'i'*19)
    batch_rows=max(1,(64*1024**2)//max(w*h*depth*4,1))
    source=driver.mem_alloc(min(batch_rows,len(raw))*w*h*depth*4)
    canonical = driver.mem_alloc(min(batch_rows,len(raw))*w*h*depth*4) if image_axes else None
    flags = driver.mem_alloc(max(1,min(batch_rows,len(raw))*max(counts)))
    image_kernel = module.get_function('tc_canonical_images') if canonical is not None else None
    if image_kernel is not None: image_kernel.prepare('PPP'+'i'*5)
    patch_kernel = module.get_function('tc_choose_patch_orientation') if patch_axes else None
    if patch_kernel is not None: patch_kernel.prepare('PP'+'i'*7)
    # One flag per image is sufficient for the whole-image prepass.
    image_flags = driver.mem_alloc(min(batch_rows,len(raw))) if canonical is not None else None
    for start in range(0,len(raw),batch_rows):
        batch=np.ascontiguousarray(raw[start:start+batch_rows])
        driver.memcpy_htod(source,batch)
        pixels = source
        if image_kernel is not None:
            image_kernel.prepared_call((min(65535,len(batch)),1,1),(256,1,1),source,canonical,image_flags,len(batch),w,h,depth,image_axes)
            pixels = canonical
        offset=0
        for scale,((pw,ph),count) in enumerate(zip(sizes,counts)):
            words=len(batch)*count*chunks
            if patch_kernel is not None:
                patch_kernel.prepared_call((min(65535,(len(batch)*count+255)//256),1,1),(256,1,1),pixels,flags,len(batch),w,h,depth,pw,ph,patch_axes)
            pack.prepared_call((min(65535,(words+255)//256),1,1),(256,1,1),pixels,
                int(output)+start*patches*chunks*4,flags,len(batch),w,h,depth,pw,ph,features,chunks,
                mw,mh,nx,ny,scale,len(sizes),patches,offset,int(len(sizes)>1),1,int(patch_kernel is not None))
            offset+=count
    # The temporary pixel buffer must stay alive until packing has finished.
    driver.Context.synchronize()
    return output
