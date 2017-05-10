import ctypes, numpy
from llvmlite import ir
import llvmlite.binding as llvm


def environment(name='nutils'):
  '''Creates a basic environment for jit-compilation. Returns a tuple (engine,
  module, stdlib), where engine is an LLVM engine object associated to the
  default compilation target, module is a module with the given name, and
  stdlib is a dict of objects (some types and functions) which form a standard
  library that can be accessed during code generation.
  '''
  target = llvm.Target.from_default_triple()
  target_machine = target.create_target_machine()
  backing_mod = llvm.parse_assembly('')
  engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
  module = ir.Module(name=name)

  stdlib = {}
  _create_types(stdlib)
  _create_libc(module, stdlib)

  return engine, module, stdlib


def NDArrayRepr(ndim):

  class NDArrayRepr(ctypes.Structure):
    # It's important that this corresponds exactly with the LLVM IR `ndarray`
    # type (see below)!
    _fields_ = [
      ('shape', ctypes.POINTER(ctypes.c_size_t)),
      ('strides', ctypes.POINTER(ctypes.c_size_t)),
      ('data', ctypes.c_void_p),
    ]

    def __init__(self, array, shape=None):
      if shape is None:
        shape = array.ctypes.shape_as(ctypes.c_size_t)
      else:
        shape = (ctypes.c_size_t * len(shape))(*shape)
        shape = ctypes.cast(ctypes.pointer(shape), ctypes.POINTER(ctypes.c_size_t))

      super(NDArrayRepr, self).__init__(
        shape, array.ctypes.strides_as(ctypes.c_size_t), array.ctypes.data,
      )
      self._array = array          # Ensure that the pointed-to data isn't freed too soon

    def __str__(self):
      return 'NDArrayRepr(ndim={}, shape=({}), strides=({}), data={})'.format(
        ndim,
        ','.join(str(c) for c in self.shape[:ndim]),
        ','.join(str(c) for c in self.strides[:ndim]),
        hex(self.data),
      )

    def array(self, dtype):
      shape = tuple(self.shape[:ndim])
      ptr = numpy.ctypeslib.ndpointer(dtype=dtype, ndim=ndim, shape=shape)(self.data)
      return numpy.ctypeslib.as_array(ptr).copy()

  return NDArrayRepr


def string_constant(string, module, bld, name):
  # Utility function for creating string constants
  type_ = ir.ArrayType(ir.IntType(8), len(string) + 1)
  variable = ir.GlobalVariable(module, type_, name)
  variable.initializer = ir.Constant(type_, list(string.encode()) + [0])
  zero = ir.Constant(ir.IntType(32), 0)
  ptr = bld.gep(variable, (zero, zero))
  return ptr


def _create_types(stdlib):
  # Create LLVM IR types that correspond to the C types on this machine.
  # This is useful for interfacing with libc.
  try:
    # Note, LLVM doesn't have unsigned integer types. Rather, the operations
    # are signed or unsigned.
    size_t = {
      ctypes.c_uint32: ir.IntType(32),
      ctypes.c_uint64: ir.IntType(64),
    }[ctypes.c_size_t]
    int_t = {
      ctypes.c_int32: ir.IntType(32),
      ctypes.c_int64: ir.IntType(64),
    }[ctypes.c_int]
  except KeyError:
    raise Exception('Unknown compilation target')

  # We use i8* as a void*
  ptr = ir.IntType(8).as_pointer()

  # A structure that corresponds to a numpy array.
  # It's important that this corresponds exactly with NDArrayRepr!
  ndarray = ir.LiteralStructType((
    int_t,                      # ndim
    size_t.as_pointer(),        # shape
    size_t.as_pointer(),        # strides
    ptr,                        # data
  ))

  stdlib.update({
    'size_t': size_t,
    'int': int_t,
    'ptr': ptr,
    'ndarray': ndarray,
  })


def _create_libc(module, stdlib):
  ptr, size_t, int_t = stdlib['ptr'], stdlib['size_t'], stdlib['int']
  stdlib.update({
    'free': ir.Function(module, ir.FunctionType(ir.VoidType(), (ptr,)), name='free'),
    'malloc': ir.Function(module, ir.FunctionType(ptr, (size_t,)), name='malloc'),
    'memcpy': ir.Function(module, ir.FunctionType(ptr, (ptr, ptr, size_t)), name='memcpy'),
    'realloc': ir.Function(module, ir.FunctionType(ptr, (ptr, size_t)), name='realloc'),
    'printf': ir.Function(module, ir.FunctionType(int_t, (ptr,), var_arg=True), name='printf'),
  })
