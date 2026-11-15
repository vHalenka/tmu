"""Validated persistent cubin cache for the cooperative Coalesced trainer."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import struct
import sys
import tempfile

from tmu.clause_bank.cuda_cooperative import CooperativeKernel, DriverError


def _read(path):
    try:
        checksum, separator, payload = path.read_bytes().partition(b"\n")
        if separator and checksum == hashlib.sha256(payload).hexdigest().encode():
            return payload
    except OSError:
        pass
    return None


def _write(path, payload):
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(hashlib.sha256(payload).hexdigest().encode() + b"\n" + payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError:
        pass  # A read-only cache must not prevent compilation or training.
    finally:
        if temporary:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def _identity(path):
    if path is None:
        return None
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns, stat.st_ino]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _dependencies(source, nvcc, options, arch):
    # Ask the actual compiler once per source/options configuration. Hash every
    # dependency on cold loads; file metadata validates digests on warm loads.
    import pycuda.compiler as compiler
    with tempfile.TemporaryDirectory() as directory:
        source_path = Path(directory) / 'kernel.cu'
        source_path.write_text(source)
        flags = list(options) + ['-arch', arch, '-I' + compiler._find_pycuda_include_path()]
        flags += ['-m64' if struct.calcsize('P') == 8 else '-m32'] if os.name == 'nt' else []
        command = [nvcc, '-M', *flags, str(source_path)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('NVCC dependency scan failed: ' + result.stderr)
        body = result.stdout.split(' : ', 1)[-1].replace('\\\n', ' ')
        # GNU make escapes spaces; nvcc uses forward slashes even on Windows.
        paths = shlex.split(body)
        return sorted({str(Path(path).resolve()) for path in paths
                       if Path(path).resolve() != source_path.resolve()})


def _header_hashes(paths, previous=None):
    # Standard filesystem invalidation: reuse a content digest only while
    # size, modification/change times and file identity remain unchanged.
    # This avoids rereading ~16 MB of toolkit headers per fresh process.
    previous = previous or {}
    records = {}
    for path in paths:
        stat = os.stat(path)
        stamp = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino]
        old = previous.get(path)
        digest = old[1] if old and old[0] == stamp else hashlib.sha256(Path(path).read_bytes()).hexdigest()
        records[path] = [stamp, digest]
    return records


def load_cooperative_kernel(source, name, signature, cache_dir=None):
    import pycuda
    import pycuda.compiler as compiler
    import pycuda.driver as cuda
    root = Path(cache_dir or os.environ.get('TMU_COALESCED_CACHE_DIR') or
                Path(tempfile.gettempdir()) / 'tm_coalesced_kernels_v1')
    nvcc = shutil.which('nvcc')
    if nvcc is None:
        raise FileNotFoundError('nvcc is required to identify the CUDA compiler')
    toolkit = Path(nvcc).resolve().parent.parent / 'version.json'
    try:
        toolkit_hash = hashlib.sha256(toolkit.read_bytes()).hexdigest()
    except OSError:
        toolkit_hash = None
    host = shutil.which(os.environ.get('NVCC_CCBIN', '') or ('cl.exe' if os.name == 'nt' else 'c++'))
    fingerprint = [_identity(nvcc), _identity(host), toolkit_hash]
    disabled = 'PYCUDA_DISABLE_CACHE' in os.environ or bool(cuda.CUDA_DEBUGGING)
    version_file = root / ('compiler-' + _digest(fingerprint) + '.meta')
    version = None if disabled else _read(version_file)
    if version is None:
        version = compiler.get_nvcc_version(nvcc).encode()
        if not disabled:
            _write(version_file, version)
    options = list(compiler.DEFAULT_NVCC_FLAGS)
    arch = 'sm_%d%d' % cuda.Context.get_device().compute_capability()
    identity = dict(format=3, source=hashlib.sha256(source.encode()).hexdigest(),
                    compiler=fingerprint, version=version.decode(), arch=arch,
                    options=options, platform=sys.platform, bits=struct.calcsize('P') * 8,
                    pycuda=pycuda.VERSION_TEXT, debugging=bool(cuda.CUDA_DEBUGGING),
                    environment={k: os.environ.get(k) for k in
                                 ['NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS', 'NVCC_CCBIN',
                                  'INCLUDE', 'CPATH', 'CPLUS_INCLUDE_PATH', 'CUDA_PATH']})
    dependencies_file = root / (_digest(identity) + '.dependencies')
    recorded = None if disabled else _read(dependencies_file)
    try:
        record = json.loads(recorded) if recorded else None
        paths = record["paths"] if record else None
        headers = _header_hashes(paths, record["headers"]) if paths is not None else None
    except (OSError, ValueError, TypeError, KeyError, IndexError):
        headers = None
    if headers is None:
        paths = _dependencies(source, nvcc, options, arch)
        headers = _header_hashes(paths)
        if not disabled:
            _write(dependencies_file, json.dumps(dict(paths=paths, headers=headers)).encode())
    identity['headers'] = sorted((path, record[1]) for path, record in headers.items())
    path = root / (_digest(identity) + '.cubin-cache')
    binary = None if disabled else _read(path)
    if binary is not None:
        try:
            return CooperativeKernel(binary, name, signature)
        except DriverError as error:
            if error.code not in (200, 209, 218, 222):
                raise
    paths = _dependencies(source, nvcc, options, arch)
    headers = _header_hashes(paths, headers)
    identity['headers'] = sorted((path, record[1]) for path, record in headers.items())
    path = root / (_digest(identity) + '.cubin-cache')
    if not disabled:
        _write(dependencies_file, json.dumps(dict(paths=paths, headers=headers)).encode())
    binary = compiler.compile(source, nvcc=nvcc, arch=arch, options=options,
                              no_extern_c=True, cache_dir=False)
    kernel = CooperativeKernel(binary, name, signature)
    if not disabled:
        _write(path, binary)
    return kernel
