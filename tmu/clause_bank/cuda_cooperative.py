"""Cooperative CUDA launches with PyCUDA-owned contexts and allocations."""
import ctypes as ct
import ctypes.util
import hashlib
import os
from functools import lru_cache

class CooperativeUnavailable(RuntimeError):
    pass

class DriverError(RuntimeError):
    def __init__(self, operation, code):
        self.code=int(code)
        super().__init__(f'{operation} failed with CUDA error {code}')

def kernel_parameters(signature, values):
    if len(signature)!=len(values):
        raise ValueError('CUDA argument count does not match the prepared signature')
    types={'P':ct.c_void_p,'i':ct.c_int32,'I':ct.c_uint32,'f':ct.c_float,'d':ct.c_double}
    owners=[types[k](float(v) if k in 'fd' else int(v or 0)) for k,v in zip(signature,values)]
    pointers=(ct.c_void_p*len(owners))(*(ct.addressof(v) for v in owners))
    return owners,pointers

class _Driver:
    def __init__(self):
        try:self.library=ct.WinDLL('nvcuda.dll') if os.name=='nt' else ct.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
        except OSError as e:raise CooperativeUnavailable(str(e)) from e
        self.functions={};self.modules={}
    def call(self,name,types,*arguments):
        function=self.functions.get(name)
        if function is None:
            try:function=getattr(self.library,name)
            except AttributeError as e:raise CooperativeUnavailable(name) from e
            function.argtypes=types;function.restype=ct.c_int;self.functions[name]=function
        result=function(*arguments)
        if result:raise DriverError(name,result)
    def supported(self):
        device=ct.c_int();self.call('cuCtxGetDevice',[ct.POINTER(ct.c_int)],ct.byref(device))
        supported=ct.c_int();self.call('cuDeviceGetAttribute',[ct.POINTER(ct.c_int),ct.c_int,ct.c_int],ct.byref(supported),95,device.value)
        return bool(supported.value)
    def load(self,binary,name):
        context=ct.c_void_p();self.call('cuCtxGetCurrent',[ct.POINTER(ct.c_void_p)],ct.byref(context))
        key=(context.value,hashlib.sha256(binary).digest())
        if key not in self.modules:
            module=ct.c_void_p();buffer=ct.create_string_buffer(binary)
            self.call('cuModuleLoadData',[ct.POINTER(ct.c_void_p),ct.c_void_p],ct.byref(module),buffer)
            self.modules[key]=module.value
        function=ct.c_void_p();self.call('cuModuleGetFunction',[ct.POINTER(ct.c_void_p),ct.c_void_p,ct.c_char_p],ct.byref(function),self.modules[key],name.encode('ascii'))
        return function.value
    def active_blocks(self,function,threads):
        blocks=ct.c_int();self.call('cuOccupancyMaxActiveBlocksPerMultiprocessor',[ct.POINTER(ct.c_int),ct.c_void_p,ct.c_int,ct.c_size_t],ct.byref(blocks),function,threads,0)
        return blocks.value
    def launch(self,function,grid,block,pointers):
        self.call('cuLaunchCooperativeKernel',[ct.c_void_p]+[ct.c_uint]*7+[ct.c_void_p,ct.POINTER(ct.c_void_p)],function,*grid,*block,0,None,pointers)

@lru_cache(maxsize=1)
def driver():return _Driver()

class CooperativeKernel:
    def __init__(self,binary,name,signature):
        self.api=driver();self.function=self.api.load(binary,name);self.signature=signature
    def active_blocks(self,threads):return self.api.active_blocks(self.function,threads)
    def launch(self,grid,block,arguments):
        owners,pointers=kernel_parameters(self.signature,arguments)
        self.api.launch(self.function,grid,block,pointers)
