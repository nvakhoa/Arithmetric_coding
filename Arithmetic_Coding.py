import ast
import random
from fractions import Fraction
from collections import defaultdict
import os
import numpy as np
import cv2
import pickle
from copy import copy


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
        for key,val in b.primes.items():
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
            buff *= pow(key,val)
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
            pass

        img = np.array(self.image)
        img = img.reshape(1,1,-1)
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

    def encoding(self):
        print('encoding...', end=' ')
        
        start = Fraction(0, 1)
        width = Fraction(1, 1)

        for code in self.codes:
            d_start, d_width = self.probs[code]
            start += d_start * width
            width *= d_width


        data = []
        data.append([hex(start.numerator), hex(start.denominator)])

        for key, val in self.counts.items():
            data.append([key, val])
            

        data = np.array(data)
        
        text_encoding = ''
        for i in data:
            text_encoding += ';{};{}'.format(i[0], i[1])

        f = open(self.path, 'w')
        f.write(text_encoding)
        f.close()
        self.start = start
        self.width = width
        print('done!!')
        return start, start + width

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

    def decoding(self, path_compression=None):
        print('decoding...', end=' ')
        path = path_compression
        if path == None:
            path = self.path
        
        f = open(path)
        data = f.read().split(';')
        f.close()

        input_fraction = Fraction(int(data[1], 0), int(data[2],0))
        
        for i in range(3, len(data), 2):
            self.counts[int(data[i],0)] = int(data[i+1],0)

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
        print('done!!')
        return ''.join([chr(code.numerator) for code in output_codes])


class Arithmetric_coding_2(object):

    def __init__(self, mod='encode', text=None):
        self.path = 'C:/Users/USER/Desktop/encoding'
        self.mod = mod
        self.text = text
        self.codes = None
        self.lenght = 0
        self.counts = defaultdict(int)

        self.start = None
        self.width = None
        self._array_start = []
        self._array_width = []

        if self.text != None:
            self.codes = [ord(char) for char in self.text] + [256]
            self.lenght = len(self.codes)
            for code in self.codes:
                self.counts[code] += 1
            self.probs = self.__build_probs__2()

    def encoding_2(self):

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

        text_encoding = ''.format(leng)
        for i in data:
            text_encoding += ';{};{}'.format(i[0], i[1])

        f = open('C:/Users/USER/Desktop/encoding.txt', 'w')
        f.write(text_encoding)
        f.close()
        self.start = start
        self.width = width
        return start, start + width

    def __build_probs__2(self):
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



def image_decoding(file, shape):
    print('decoding...',end='')
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


path = 'D:/source/PythonProject/Text_classification/train/53637 sci.electronics'
path_compression = 'C:/Users/USER/Desktop/encoding.txt'
f = open(path, 'r')
text = f.read()

encode = Arithmetric_coding(
    path_compression=path_compression, pathdir=path, mod='encode')
encode.encoding()
decode = Arithmetric_coding(mod='decode')
text_decoding = decode.decoding(path_compression)

print(text_decoding)
