"""
Module for reading and writing N-body snapshots in the NEMO binary format.
The code for snapshot handling is hacked from AMUSE and reworked into a standalone module.
"""
import sys, array, numpy

__all__ = ['NemoFile']

# save the original built-in function for opening a file; the name will later be reassigned to a custom open function
_builtins_open = open

class OrderedMultiDictionary(object):
    """A dictionary that keeps the keys in the dictionary in order and can store
    multiple items per key

    Ordered multi dictionaries remember the order that items were inserted
    and can store multiple values per key.  When iterating over an ordered dictionary,
    the values are returned in the order their keys were first added.

    >>> d = OrderedMultiDictionary()
    >>> d["first"] = 0
    >>> d["second"] = 1
    >>> d["first"] = 2
    >>> [x for x in d]
    [0, 1, 2]
    >>> print d["first"]
    [0, 2]
    >>> print d["second"]
    [1]
    """

    def __init__(self):
        self.mapping = {}
        self.orderedKeys = []

    def __setitem__(self, key, value):
        if not key in self.mapping:
            self.mapping[key] = []
        self.mapping[key].append(value)
        self.orderedKeys.append(
            (
                key,
                len(self.mapping[key]) - 1,
            )
        )

    def __getitem__(self, key):
        return self.mapping[key]

    def __contains__(self, key):
        return key in self.mapping

    def __iter__(self):
        return list(self.values())

    def __len__(self):
        return len(self.orderedKeys)

    def __str__(self):
        result = "OrderedDictionary({"
        elements = []
        for x, index in self.orderedKeys:
            elements.append(repr(x) + ":" + repr(self[x][index]))
        result += ", ".join(elements)
        result += "})"
        return result

    def __repr__(self):
        return str(self)

    def __getattr__(self, key):
        return self.mapping[key]

    def keys(self):
        return [x for x, index in self.orderedKeys]

    def values(self):
        for x, index in self.orderedKeys:
            yield self.mapping[x][index]


# copied from 'six' module to avoid adding it as a dependency
def add_metaclass(metaclass):
    """Class decorator for creating a class with a metaclass."""
    def wrapper(cls):
        orig_vars = cls.__dict__.copy()
        slots = orig_vars.get('__slots__')
        if slots is not None:
            if isinstance(slots, str):
                slots = [slots]
            for slots_var in slots:
                orig_vars.pop(slots_var)
        orig_vars.pop('__dict__', None)
        orig_vars.pop('__weakref__', None)
        if hasattr(cls, '__qualname__'):
            orig_vars['__qualname__'] = cls.__qualname__
        return metaclass(cls.__name__, cls.__bases__, orig_vars)
    return wrapper


class NemoItemType(type):
    mapping = {}

    def __new__(metaclass, name, bases, dict):
        if 'datatype' in dict:
            if not dict['datatype'] is None:
                dict['datatype'] = numpy.dtype(dict['datatype'])
        result = type.__new__(metaclass, name, bases, dict)
        if 'typecharacter' in dict:
            metaclass.mapping[dict['typecharacter']] = result
        return result

    @classmethod
    def new_item(metaclass, typecharacter, tagstring, dimensions, mustswap=False):
        return metaclass.mapping[typecharacter](tagstring, dimensions, mustswap=mustswap)


@add_metaclass(NemoItemType)
class NemoItem(object):

    def __init__(self, tagstring, dimensions=[1], data=None, mustswap=False):
        self.tagstring = tagstring
        self.dimensions = dimensions 
        self.mustswap = mustswap
        self.data = data

    def is_plural(self):
        if len(self.dimensions) == 1 and self.dimensions[0] <= 1:
            return False
        else:
            return True

    def read(self, nemofile):
        if self.datatype is None:
            pass
        else:
            self.data = nemofile.read_fixed_array(self.datatype, numpy.prod(self.dimensions))
        self.postprocess()

    def write(self, nemofile):
        if self.datatype is None:
            pass
        else:
            nemofile.write_fixed_array(self.preprocess(), self.datatype)

    def postprocess(self):
        if self.mustswap:
            self.data.byteswap()

    def preprocess(self):
        return self.data

    def isEndOfSet(self):
        return False

    def isEndOfHistory(self):
        return False

    def __str__(self):
        return 'nemoitem({0},{1})'.format(self.tagstring, self.dimensions)

    def __repr__(self):
        return '<{0!s} {1},{2}>'.format(type(self), self.tagstring, self.dimensions)


class AnyItem(NemoItem):
    """anything at all"""
    typecharacter = "a"
    datatype = numpy.byte


class CharItem(NemoItem):
    """printable chars"""
    typecharacter = "c"
    datatype = "c"

    def postprocess(self):
        if sys.version_info.major == 2:
            self.data = ''.join(self.data[:-1])
        else:
            self.data = self.data[:-1].tobytes().decode('latin_1')

    def preprocess(self):
        result = numpy.array(list(self.data), "c")
        result = numpy.append(result, b'\x00')
        return result


class ByteItem(NemoItem):
    """unprintable chars"""
    typecharacter = "b"
    datatype = numpy.byte

    def postprocess(self):
        self.data = self.data.reshape(self.dimensions)


class ShortItem(NemoItem):
    """  short integers """
    typecharacter = "s"
    datatype = numpy.int16

    def postprocess(self):
        self.data = self.data.reshape(self.dimensions)


class IntItem(NemoItem):
    """  standard integers """
    typecharacter = "i"
    datatype = numpy.int32

    def postprocess(self):
        self.data = self.data.reshape(self.dimensions)


class LongItem(NemoItem):
    """  long integers """
    typecharacter = "l"
    datatype = numpy.int64

    def postprocess(self):
        self.data = self.data.reshape(self.dimensions)


class HalfpItem(NemoItem):
    """  half precision floating """
    typecharacter = "h"
    datatype = numpy.float16


class FloatItem(NemoItem):
    """  short floating """
    typecharacter = "f"
    datatype = numpy.float32

    def postprocess(self):
        self.data = self.data.reshape(self.dimensions)


class DoubleItem(NemoItem):
    """  long floating """
    typecharacter = "d"
    datatype = numpy.float64

    def postprocess(self):
        self.data = self.data.reshape(self.dimensions)


class SetItem(NemoItem):
    """  begin compound item """
    typecharacter = "("
    datatype = None

    def __init__(self, tagstring, dimensions=[1], data=None, mustswap=False):
        if data is None:
            data = OrderedMultiDictionary()
        NemoItem.__init__(self, tagstring, dimensions, data, mustswap)

    def read(self, nemofile):
        self.data = OrderedMultiDictionary()
        subitem = nemofile.read_item()
        while not subitem.isEndOfSet():
            self.data[subitem.tagstring] = subitem
            subitem = nemofile.read_item()

    def write(self, nemofile):
        for x in self.data.values():
            nemofile.write_item(x)
        nemofile.write_item(TesItem(self.tagstring, [1]))

    def add_item(self, item):
        self.data[item.tagstring] = item


class TesItem(NemoItem):
    """  end of compound item """
    typecharacter = ")"
    datatype = None

    def isEndOfSet(self):
        return True

    def read(self, file):
        pass


class StoryItem(NemoItem):
    """  begin of a story item (see starlab) """
    typecharacter = "["
    datatype = None

    def read(self, nemofile):
        self.data = OrderedMultiDictionary()
        subitem = nemofile.read_item()
        while not subitem.isEndOfHistory():
            self.data[subitem.tagstring] = subitem
            subitem = nemofile.read_item()
        print('read story: %s' % self.data)

    def write(self, nemofile):
        for x in self.data.values():
            nemofile.write_item(x)
        nemofile.write_item(YrotsItem(self.tagstring, [1]))


class YrotsItem(NemoItem):
    """  end of a story item (see starlab) """
    typecharacter = "]"
    datatype = None

    def isEndOfHistory(self):
        return True


class NemoFile(object):
    """
    Class for reading and writing NEMO binary snapshots.
    Like an ordinary file, it can be used as a context manager and/or iterator:

    # open a file for reading, iterate over multiple snapshots in the file
    for snap in NemoFile('input.snap'):
        print(snap['Time'], len(snap['Position']))

    # open a file for writing using a context manager, appending to the end
    # if it exists, write a single snapshot
    with open('output.snap', 'a') as file:
        file.write(dict(Time=1.0, Position=[[1, 2, 3]]))
    """

    def __init__(self, filename, mode='r'):
        """
        Create a NemoFile object for reading or writing, depending on mode.
        Arguments:
          filename:  file name
          mode:  access mode, must be one of the following:
            'r' - read (default);
            'x' - write to a new file, fail if the file already exists;
            'w' - write, overwrite the file if it already exists;
            'a' - write, append to the end of the file if it already exists.
        """
        if mode not in ('r','x','w','a'):
            raise RuntimeError("mode should be one of the following: 'r', 'x', 'w', 'a'")
        self.file = _builtins_open(filename, mode+'b')
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def close(self):
        self.file.close()

    def __iter__(self):
        return self

    SingMagic = 0x0992
    PlurMagic = 0x0b92
    reversed_SingMagic = 0x9209
    reversed_PlurMagic = 0x920b

    def _byteswap(self, value, type='H'):
        x = array.array('H', [value])
        x.byteswap()
        return x[0]

    def read_magic_number(self):
        nbytes = 2
        bytes = self.file.read(nbytes)
        if not bytes or len(bytes) < nbytes:
            return None
        return array.array('h', bytes)[0]

    def read_array(self, typetag):
        result = array.array(typetag)
        must_loop = True
        while must_loop:
            block = self.file.read(result.itemsize)
            if sys.version_info.major == 2:
                result.fromstring(block)
            else:
                result.frombytes(block)
            must_loop = result[-1] != 0

        result.pop()
        return result

    def read_string(self):
        if sys.version_info.major == 2:
            return self.read_array('b').tostring()
        else:
            return self.read_array('b').tobytes().decode('latin_1')

    def read_fixed_array(self, datatype, count):
        bytes = self.file.read(int(datatype.itemsize * count))
        if sys.version_info.major == 2:
            return numpy.fromstring(bytes, dtype=datatype,)
        else:
            return numpy.frombuffer(bytes, dtype=datatype,)

    def get_item_header(self):

        magic_number = self.read_magic_number()
        if magic_number is None:
            return (None, None, None, None)
        mustswap = False
        if magic_number == self.reversed_SingMagic:
            magic_number == self.SingMagic
            mustswap = True
        elif magic_number == self.reversed_PlurMagic:
            magic_number == self.PlurMagic
            mustswap = True

        if not (magic_number == self.SingMagic or magic_number == self.PlurMagic):
            raise RuntimeError("Item does not have a valid header")

        typecharacter = self.read_string()
        if not typecharacter == TesItem.typecharacter:
            tagstring = self.read_string()
        else:
            tagstring = ''
        if magic_number == self.PlurMagic:
            dim = self.read_array('i')
            if mustswap:
                dim.byteswap()
            dim = dim.tolist()
        else:
            dim = [1]

        return (typecharacter, tagstring, dim, mustswap)

    def read_item(self):
        typecharacter, tagstring, dim, mustswap = self.get_item_header()
        if typecharacter is None:
            return None
        result = NemoItemType.new_item(typecharacter, tagstring, dim, mustswap)
        result.read(self)
        return result


    def write_magic_number(self, is_plural):
        if is_plural:
            magic_number = self.PlurMagic
        else:
            magic_number = self.SingMagic

        x = array.array('h', [magic_number])
        self.file.write(x.tostring() if sys.version_info.major == 2 else x.tobytes())

    def write_array(self, typetag, data):
        x = array.array(typetag, data)
        x.append(0)
        self.file.write(x.tostring() if sys.version_info.major == 2 else x.tobytes())

    def write_string(self, string):
        return self.write_array('b', string if sys.version_info.major == 2 else string.encode('latin_1'))

    def write_item_header(self, item):
        self.write_magic_number(item.is_plural())
        self.write_string(item.typecharacter)
        if not item.typecharacter == TesItem.typecharacter:
            self.write_string(item.tagstring)
        if item.is_plural():
            self.write_array('i', item.dimensions)

    def write_item(self, item):
        self.write_item_header(item)
        item.write(self)

    def write_fixed_array(self, data, datatype):
        temp = numpy.array(data, dtype=datatype)
        self.file.write(temp.tostring() if sys.version_info.major == 2 else temp.tobytes())

    def read(self):
        """
        Read the next snapshot from the NEMO file and return it as a dictionary,
        typically containing the following items (although none of them are guaranteed to be present):
          Time  (float)
          Position  (2d array of shape Nx3)
          Velocity  (2d array of shape Nx3)
          Mass  (1d array of length N)
        If there are no more snapshots in the file, return None.
        """
        while True:
            item = self.read_item()
            if item is None:
                return
            if item.tagstring != 'SnapShot':
                continue
            result = {}
            if 'Parameters' in item.data:
                if 'Time' in item.data['Parameters'][0].data:
                    result['Time'] = item.data['Parameters'][0].data['Time'][0].data[0]
            if 'Particles' in item.data:
                for par in item.data['Particles'][0].data.keys():
                    if par == 'CoordSystem':
                        continue  # skip
                    elif par == 'PhaseSpace':
                        posvel = item.data['Particles'][0].data[par][0].data
                        result['Position'] = posvel[:,0]
                        result['Velocity'] = posvel[:,1]
                    else:
                        result[par] = item.data['Particles'][0].data[par][0].data
            return result

    def __next__(self):
        result = self.read()
        if result is None:
            raise StopIteration
        return result

    next = __next__  # alias for Python 2.x

    def write(self, snapshot):
        """
        Write or append a snapshot into a NEMO file.
        The snapshot must be a dictionary typically containing the following items,
        although none of them are strictly required:
          Time  (float)
          Position  (2d array of shape Nx3)
          Velocity  (2d array of shape Nx3)
          Mass  (1d array of length N)
        """
        if not isinstance(snapshot, dict):
            raise ValueError('Argument must be a dictionary')
        item = SetItem('SnapShot')
        parameters_item = SetItem('Parameters')
        particles_item = SetItem('Particles')
        nbody = None
        for key in snapshot:
            if key == 'Time':
                parameters_item.add_item(DoubleItem('Time', data=float(snapshot['Time'])))
            else:
                arr = numpy.asanyarray(snapshot[key])
                if   arr.dtype == numpy.int8   : Item = ByteItem
                elif arr.dtype == numpy.int16  : Item = ShortItem
                elif arr.dtype == numpy.int32  : Item = IntItem
                elif arr.dtype == numpy.int64  : Item = LongItem
                elif arr.dtype == numpy.float16: Item = HalfpItem
                elif arr.dtype == numpy.float32: Item = FloatItem
                elif arr.dtype == numpy.float64: Item = DoubleItem
                else:
                    raise TypeError('Invalid dtype: %s' % arr.dtype)
                particles_item.add_item(Item(key, dimensions=arr.shape, data=arr))
                if nbody is None:
                    nbody = len(arr)
                elif nbody != len(arr):
                    raise ValueError('Array lengths should be identical')
        parameters_item.add_item(IntItem('Nobj', data=nbody))
        item.add_item(parameters_item)
        item.add_item(particles_item)
        self.write_item(item)
        self.file.flush()

# convenience alias
def open(filename, mode='r'):
    return NemoFile(filename, mode)

open.__doc__ = NemoFile.__init__.__doc__
