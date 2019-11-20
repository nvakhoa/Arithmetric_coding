# author: Nguyen Vu Anh Khoa - 16521511
import ast
import random
from fractions import Fraction
from collections import defaultdict
import os
import numpy as np
import cv2
from copy import copy
from time import time
from math import log
import struct
import contextlib
import sys

# references
r1 = 'https://www.theunterminatedstring.com/python-bits-and-bytes/'
r2 = 'https://github.com/nayuki/Reference-arithmetic-coding'
r3 = 'wiki'
r4 = 'https://www.youtube.com/watch?v=7vfqhoJVwuc'
r5 = 'https://github.com/gw-c/arith'

# cmd: Arithmetic_Coding.py [version] [mode] [file1] [file(s)2]
# cmd:    python D:\source\PythonProject\Image\compression\Arithmetic_Coding.py 3 encode C:\Users\USER\Desktop\testing\1.txt C:\Users\USER\Desktop\testing\encode


class Prime():

    def __init__(self, interger=None):
        self.number = interger
        self.primes = dict()
        if not interger == None:
            self.primes = self._arrayPrime()

    def _get_(self):
        return self.primes

    def _arrayPrime(self):
        number = self.number
        if number == None:
            print("number is None to AddPrime")
            return []

        i = 2
        arr = dict()
        while i * i < number:
            k = 0
            while number % i == 0:
                number = number / i
                k += 1
            if k > 0:
                arr[i] = k
            i = i + 1
        arr[int(number)] = 1
        return arr

    def _printPrime(self):
        pass
        print(self.primes)

    def add(self, b):
        """
        eg: (2^2 * 3^2) + (2 * 3) =  (2 * 3) * ( 2 * 3 + 1 )  = (2 * 3 * 7)
        """
        a = copy(self)
        b = copy(b)
        list_key = list(a.primes.keys())
        list_key.extend(list(b.primes.keys()))
        list_key = set(list_key)

        result = Prime()

        for key in list_key:

            val_a = a.primes.get(key, 0)
            val_b = b.primes.get(key, 0)
            min_val = min(val_b, val_a)
            val_b -= min_val
            val_a -= min_val

            a.primes[key] = val_a

            b.primes[key] = val_b

            if min_val > 0:
                result.primes[key] = min_val

        buff = a._toInt() + b._toInt()
        result._upgrade(Prime(buff))

        return result

    def _upgrade(self, b):
        for key, val in b.primes.items():
            self.primes[key] = self.primes.get(key, 0) + val

    def mul(self, b):
        """
        eg: 2^3 * 2^6 = 2^9
        """

        mul = copy(self)
        for key, val in b.items():
            mul.primes[key] = self.primes.get(key, 0) + val
        return mul

    def items(self):
        return self.primes.items()

    def _toInt(self):
        buff = 1
        for key, val in self.items():
            buff *= pow(key, val)
        return buff


class Fraction_prime():
    def __init__(self, a, b):
        self.numer = a
        self.deno = b

    def mulFractPrime(self, FractionPrime):
        result = copy(self)
        result.numer = self.numer.mul(FractionPrime.numer)
        result.denor = self.deno.mul(FractionPrime.deno)
        return result

    def addFractPrime(self, FractionPrime):
        result = copy(self)
        result.numer = self.numer.add(FractionPrime.numer)
        result.denor = self.deno.add(FractionPrime.deno)
        return result

    def toInt(self):
        return (self.numer._toInt(), self.deno._toInt())


class Arithmetric_coding_image(object):

    def __init__(self, path):
        self.image = cv2.imread(path)
        self.shape = (self.image.shape[:2])
        self.codes = None
        self.imge2String()
        self.counts = defaultdict(int)
        self.lenght = len(self.codes)
        for code in self.codes:
            self.counts[code] += 1
        self.probs = self.__build_probs__()

    def imge2String(self):
        if self.image.shape[2] != 3:
            print('image shape 2 != 3')

        img = np.array(self.image)
        img = img.reshape(1, 1, -1)
        self.codes = img[0][0][:]

    def __build_probs__(self):
        probs = dict()
        cumulative_count = 0
        for char in self.counts.keys():
            probs[char] = Fraction(
                cumulative_count, self.lenght), Fraction(self.counts[char], self.lenght)
            cumulative_count += self.counts[char]

        return probs
        # for i in  img[0][0]:
        #     self.codes += '{} '.format(i)

    def encoding(self):
        print('encoding...', end='')

        start = Fraction(0, 1)
        width = Fraction(1, 1)
        for code in self.codes:
            d_start, d_width = self.probs[code]
            start += d_start * width
            width *= d_width

        text_encoding = '{}'.format(start)
        new_dict = ''
        for key, val in self.probs.items():
            new_dict += '{} {} {} {} {} '.format(key, val[0].numerator, val[0].denominator,
                                                 val[1].numerator, val[1].denominator)
        text_encoding += '\n' + new_dict

        f = open('C:/Users/USER/Desktop/image_encoding.txt', 'w')
        f.write(text_encoding)
        f.close()
        print('done!')
        return start, start + width


class Arithmetric_coding(object):

    def __init__(self, path_compression=None, pathdir=None, mod='encode'):
        self.path = path_compression
        self.pathDir = pathdir
        self.mod = mod
        self.text = None
        self.codes = None
        self.lenght = 0
        self.counts = defaultdict(int)

        self.start = None
        self.width = None
        self.probs = dict()
        self._array_start = []
        self._array_width = []

        if self.pathDir != None:
            f = open(self.pathDir, 'r')
            self.text = f.read()
            f.close()

            self.codes = [ord(char) for char in self.text] + [256]
            self.lenght = len(self.codes)
            for code in self.codes:
                self.counts[code] += 1
            self.probs = self.__build_probs__()

    def __build_probs__(self):

        cumulative_count = 0
        probs = dict()
        if self.mod == 'encode':
            for char in self.counts.keys():
                probs[char] = Fraction(
                    cumulative_count, self.lenght), Fraction(self.counts[char], self.lenght)
                cumulative_count += self.counts[char]
        else:
            if self.lenght == 0:
                for cnt in self.counts.values():
                    self.lenght += cnt
            for char in self.counts.keys():
                probs[char] = Fraction(
                    cumulative_count, self.lenght), Fraction(self.counts[char], self.lenght)
                cumulative_count += self.counts[char]

        return probs

    def encoding(self):
        name1 = '.txt'
        start = Fraction(0, 1)
        width = Fraction(1, 1)

        for code in self.codes:
            d_start, d_width = self.probs[code]
            start += d_start * width
            width *= d_width

        data = []
        data.append([hex(start.numerator), hex(start.denominator), ])

        for key, val in self.counts.items():
            data.append([key, val])

        data = np.array(data)

        text_encoding = ''
        for i in data:
            text_encoding += ';{};{}'.format(i[0], i[1])

        f = open(self.path + name1, 'w')
        f.write(text_encoding)
        f.close()
        self.size_compression = os.path.getsize(self.path + name1)
        self.start = start
        self.width = width

        return start, start + width

    def decoding(self, path_compression=None):
        name1 = '.txt'
        path = path_compression
        if path == None:
            path = self.path

        f = open(path + name1)
        data = f.read().split(';')
        f.close()

        for i in range(3, len(data), 2):
            self.counts[int(data[i], 0)] = int(data[i+1], 0)

        self.probs = self.__build_probs__()
        input_fraction = Fraction(int(data[1], 16), int(data[2], 16))

        output_codes = []
        code = 257
        while code != 256:
            for code, (start, width) in self.probs.items():
                current = input_fraction - start
                if 0 <= current < width:
                    input_fraction = (current) / width
                    if code < 256:
                        output_codes.append(code)
                    break

        return ''.join([chr(code.numerator) for code in output_codes])


class Arithmetric_coding_2(Arithmetric_coding):

    def encoding_(self):

        print('encoding...')

        start = Fraction_prime(Prime(0), Prime(1))
        width = Fraction_prime(Prime(1), Prime(1))

        for code in self.codes:
            d_start, d_width = self.probs[code]
            start = start.addFractPrime(width.mulFractPrime(d_start))

            width = width.mulFractPrime(d_width)

        # zip_code =
        # (N - 1) // f2 + 1
        data = []

        #text_encoding = '{:.9999f}'.format(Decimal(start.numerator)/Decimal(start.denominator))
        # new_dict = '

        for key, val in start.numer.items():
            data.append([key, val])

        for key, val in start.deno.items():
            data.append([key, val])

        for key, val in self.counts.items():
            data.append([key, val])

            # new_dict += '{} {} {} {} {} '.format(key, val[0].numerator, val[0].denominator,
            #                                     val[1].numerator, val[1].denominator)
        #text_encoding += '\n' + new_dict

        leng = (len(start.numer.items()), len(
            start.deno.items()), len(self.counts.items()))

        data = np.array(data)
        # np.save(self.path, data)

        text_encoding = "{}".format(leng)
        for i in data:
            text_encoding += ';{};{}'.format(i[0], i[1])

        f = open('C:/Users/USER/Desktop/encoding.txt', 'w')
        f.write(text_encoding)
        f.close()
        self.start = start
        self.width = width
        return start, start + width

    def __build_probs__(self):
        probs = dict()
        cumulative_count = 0

        lenght = Prime(self.lenght)
        if self.mod == 'encode':
            # self._array_width.append(arrayPrime(self.lenght))

            for char in self.counts.keys():
                start = Prime(cumulative_count)
                width = Prime(self.counts[char])
                probs[char] = Fraction_prime(
                    start, lenght), Fraction_prime(width, lenght)
                # self._array_start.append(arrayPrime(cumulative_count))
                cumulative_count += self.counts[char]
        else:
            if self.lenght == 0:
                for cnt in self.counts.values:
                    self.lenght += cnt
        return probs

    def decoding(self, file=None):
        print('decoding...')
        path = file
        if file == None:
            path = self.path

        f = open(path)
        data = f.read().split(';')
        f.close()

        input_fraction = Fraction(int(data[1], 0), int(data[2], 0))

        for i in range(3, len(data), 2):
            self.counts[int(data[i], 0)] = int(data[i+1], 0)

        self.__build_probs__()

        output_codes = []
        code = 257
        while code != 256:
            for code, (start, width) in self.probs.items():
                current = input_fraction - start
                if 0 <= current < width:
                    input_fraction = (current) / width
                    if code < 256:
                        output_codes.append(code)
                    break

        return ''.join([chr(code.numerator) for code in output_codes])


class Arithmetric_coding_3(Arithmetric_coding):

    def encoding(self):

        start = Fraction(0, 1)
        width = Fraction(1, 1)

        for code in self.codes:
            d_start, d_width = self.probs[code]
            start += d_start * width
            width *= d_width

        self.start = start
        self.width = width
        self.wirteFile()

        return start, start + width

    def decoding(self, path_compression=None, Bstart=None):

        if path_compression != None:
            self.path = path_compression
        self.readFile()

        input_fraction = self.number

        if Bstart != None:
            input_fraction = Bits2Fraction(Bstart)

        for i in range(1, len(self.data), 2):
            self.counts[int(self.data[i], 0)] = int(self.data[i+1], 0)

        self.probs = self.__build_probs__()

        output_codes = []
        code = 257
        while code != 256:
            for code, (start, width) in self.probs.items():
                current = input_fraction - start
                if 0 <= current < width:
                    input_fraction = (current) / width

                    if code < 256:
                        output_codes.append(code)
                    break

        return ''.join([chr(code.numerator) for code in output_codes])

    def readFile(self):
        name1 = '.txt'
        name2 = '.bnr'

        with open(self.path + name1) as f:
            self.data = f.read().split(';')
        self.data.extend(['256', '1'])
        self.Bstart = readBytes(self.path + name2)

        self.number = Bits2Fraction(self.Bstart)

    def wirteFile(self):
        name1 = '.txt'
        name2 = '.bnr'

        data = []
        for key, val in self.counts.items():
            data.append([key, val])

        data = np.array(data)

        text_encoding = ''

        for i in data[:-1]:
            text_encoding += ';{};{}'.format(int(i[0]), int(i[1]))

        with open(self.path + name1, 'w') as f:
            f.write(text_encoding)

        self.Bstart = Fraction2Bits(self.start, self.width)
        self.size_compression = os.path.getsize(
            self.path + name1) + os.path.getsize(self.path + name2)
        writeBytes(self.path + name2, self.Bstart)


def image_decoding(file, shape):
    print('decoding...', end='')
    # image_encoding
    f = open(file, 'r')

    input_fraction = Fraction(f.readline().split(' ')[0])
    raw_probs = f.readline().split(' ')
    f.close()

    raw_probs = [int(char) for char in raw_probs if char != '']
    probs = dict()

    i = 0
    while i < len(raw_probs):
        probs[raw_probs[i]] = Fraction(
            raw_probs[i+1], raw_probs[i+2]), Fraction(raw_probs[i+3], raw_probs[i+4])
        i += 5

    output_codes = []
    code = 257
    lenght = shape[0] * shape[1]
    while len(output_codes) < lenght:
        for code, (start, width) in probs.items():
            current = input_fraction - start
            if 0 <= current < width:
                input_fraction = (current) / width
                if code < 256:
                    output_codes.append(code)
                break

    return ''.join([chr(code) for code in output_codes])


def check(start, end, Bstart, Bend):
    assert Bstart < Bend

    if Bstart >= start and Bend < end:
        return True


def Fraction2Bits(start, width):
    bits = ''
    bitstart = ''
    bitsend = ''
    end = start + width

    Bstart = Fraction(0, 1)
    Bwidth = Fraction(1, 2)
    Bend = Fraction(1, 1)
    mid = Fraction(1, 2)

    while not check(start, end, Bstart, Bend):

        if mid > start:
            if mid > end:
                bits += '0'
                mid -= Bwidth*Fraction(1, 2)
            else:
                bits += '1'
                if Bstart < start:
                    Bstart = mid
                    bitstart = bits
                mid += Bwidth*Fraction(1, 2)
        else:
            bits += '1'
            if Bstart < start:
                Bstart = mid
                bitstart = bits
            mid += Bwidth*Fraction(1, 2)

        Bwidth *= Fraction(1, 2)
        Bend = Bstart + Bwidth
        bitsend = bits

    return bitstart


def writeBytes(pathFile, bits):
    bits = bits
    with open(pathFile, "wb") as f:
        for i in range(0, len(bits), 8):
            t = int(bits[i:i + 8][::-1], 2).to_bytes(1, 'little')
            f.write(t)


def readBytes(pathFile):
    bits = ''
    with open(pathFile, 'rb') as f:

        chars = f.read()
        # for i in range(0,len(chars),8):
        interger = int.from_bytes(chars, 'little')
        bits += '{0:08b}'.format(interger)[::-1]
    return bits


def Bits2Fraction(bits):
    result = 0
    for i in range(len(bits)):
        if bits[i] == '1':
            result += Fraction(1, 2**(i+1))
    return result


def compare(arr1, arr2):
    len1 = len(arr1)
    len2 = len(arr2)
    lenght = len1 if len1 < len2 else len2

    # for i in range(lenght):
    if arr1[:lenght] != arr2[:lenght]:
        # # print('arr1 :',arr1[i-1:i+20])
        # # print('arr2 :',arr2[i-1:i+20])
        return False
    return True


def main():
    sys.argv = sys.argv[1:]

    assert len(sys.argv) == 4, '{} != 4'.format(len(sys.argv))
    version = sys.argv[0]
    mode = sys.argv[1]
    assert version in ['1', '2', '3']
    assert mode in ['encode', 'decode']

    if version == '1':
        ArthCode = Arithmetric_coding
    elif version == '2':
        ArthCode = Arithmetric_coding_2
    elif version == '3':
        ArthCode = Arithmetric_coding_3
    else:
        print("version {} is not in my versions".format(version))

    path_file = sys.argv[2]
    path_compression = sys.argv[3]

    if mode == 'encode':
        print('encoding...', end=' ')
        tic = time()
        encode = ArthCode(
            path_compression=path_compression, pathdir=path_file, mod='encode')
        encode.encoding()
        print('done!! {0:8.2f} s'.format(time() - tic))
        size_root = os.path.getsize(path_file)
        size_compression = encode.size_compression
        print('size file: {} b, size compress: {} b,  file/compress:{:8.2f}'.format(
            size_root, size_compression, size_root/size_compression))

    elif mode == 'decode':

        print('decoding...', end=' ')
        tic = time()

        decode = ArthCode(mod='decode', path_compression=path_compression)
        text_decoding = decode.decoding()

        print('done!! {0:8.2f} s'.format(time() - tic))

        path_file = path_file[:-4] + '_decompress.txt'

        with open(path_file, 'w+') as f:
            f.write(text_decoding)
        print('Saved data decompression in {}'.format(path_file))


if __name__ == "__main__":
    main()
