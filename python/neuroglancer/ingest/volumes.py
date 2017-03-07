import numpy as np
import requests
from base import Storage
import re
import blosc

class Volume(object):

    def __getitem__(self, slices):
        """
        Asumes x,y,z coordinates
        """
        raise NotImplemented

    @property
    def shape(self):
        """
        Asumes x,y,z coordinates
        """
        return self._shape

    @property
    def data_type(self):
        """
        Data type of the voxels in this volume
        """

        return self._data_type
    

    @property
    def layer_type(self):
        """
        Either segmentation or image
        """
        return self._layer_type

    @property
    def mesh(self):
        """
        Return True if mesh is desired
        """
        return self._mesh

    @property
    def resolution(self):
        """
        Size of voxels in nanometers
        """
        return self._resolution

    @property
    def underlying(self):
        """
        Size of the underlying chunks
        """
        return self._underlying

    @property
    def num_channels(self):
        if len(self.shape) == 3:
            return 1
        elif len(self.shape) == 4:
            return self.shape[0]
        else:
            raise Exception('Wrong shape')  
    
class HDF5Volume(Volume):

    def __init__(self, path):
        import h5py
        self._f = h5py.File(path, 'r')
        self._data = self._f['main']
        self._layer_type = 'image'
        self._mesh = False
        self._resolution = [4,4,40]
        self._underlying = self.shape
        self._data_type = self._f['main'].dtype

    @property
    def shape(self):
        return self._data.shape[::-1]


    def __getitem__(self, slices):
        """
        Asumes x,y,z coordinates
        """
        return np.swapaxes(self._data.__getitem__(slices[::-1]),0,2)

    def __del__(self):
        self._f.close()

class FakeVolume(Volume):

    def __init__(self):
        arr = np.ones(shape=(127,127,127),dtype=np.uint32)
        self._data = np.pad(arr, 1, 'constant')
        self._layer_type = 'image'
        self._mesh = False
        self._resolution = [6,6,30]
        self._underlying = self.shape
        self._data_type = self._data.dtype

    @property
    def shape(self):
        return self._data.shape


    def __getitem__(self, slices):
        """
        Asumes x,y,z coordinates
        """
        return self._data.__getitem__(slices)

class DVIDVolume(Volume):

    def __init__(self):
        self._info = requests.get('http://seung-titan01.pni.princeton.edu:8000/api/node/5d7b0fea4b674a1ea48020f1abaaf009/tiles4/info').json()
        self._resolution = self._info['Extended']['Levels']['0']['Resolution']
        self._shape = self._info['Extended']['MaxTileCoord']
        self._underlying = self._info['Extended']['Levels']['0']['TileSize']
        self._layer_type = 'image'
        self._mesh = False
        self._data_type = 'uint8'

    def __getitem__(self, slices):
        x, y, z = slices
        x_size = x.stop - x.start; x_min = x.start
        y_size = y.stop - y.start; y_min = y.start
        z_size = z.stop - z.start; z_min = z.start

        url = "{api}/node/{UUID}/{dataname}/raw/{dims}/{size}/{offset}/nd".format(
            api="http://seung-titan01.pni.princeton.edu:8000/api",
            UUID="5d7b0fea4b674a1ea48020f1abaaf009",
            dataname="grayscale",
            dims="0_1_2",
            size="_".join(map(str,[x_size,y_size,z_size])),
            offset="_".join(map(str,[x_min, y_min, z_min])),
            )

        return np.swapaxes(np.fromstring(requests.get(url).content , np.uint8).reshape(z_size,y_size,x_size),0,2)


class Pinky(Volume):

    def __init__(self):
        self._resolution = [4,4,40]
        self._layer_type = 'image'
        self._mesh = False
        self._underlying = [2048,2048,64]
        self._data_type = 'uint8'
        dataset_name = 'pinky_v0'
        layer_name = 'image'
        abs_x_min = abs_y_min = abs_z_min = float('inf')
        abs_x_max = abs_y_max = abs_z_max = 0
        storage = Storage(dataset_name=dataset_name, layer_name=layer_name, compress=False)
        self._bucket = storage.client.get_bucket('seunglab')
        i = 0 
        for blob in self._bucket.list_blobs(prefix='pinky40/images/'):
            i+=1
            name = blob.name.split('/')[-1]
            if name == 'config.json':
                continue

            x_min, x_max, y_min, y_max , z_min, z_max = re.match(r'^(\d+):(\d+)_(\d+):(\d+)_(\d+):(\d+)$', name).groups()
            abs_x_min = min(int(x_min), abs_x_min)
            abs_y_min = min(int(y_min), abs_y_min)
            abs_z_min = min(int(z_min), abs_z_min)
            abs_x_max = max(int(x_max), abs_x_max)
            abs_y_max = max(int(y_max), abs_y_max)
            abs_z_max = max(int(z_max), abs_z_max)

        self._shape = [abs_x_max-abs_x_min+1,
                       abs_y_max-abs_y_min+1,
                       abs_z_max-abs_z_min+1]

        print('there are {} chunks'.format(i))
        print (self._shape)
        self._offset = [abs_x_min-1, abs_y_min-1, abs_z_min-1]
        print (self._offset)

    def __getitem__(self, slices):
        (x,y,z) = slices
        path = 'pinky40/images/{}:{}_{}:{}_{}:{}'.format(
            self._offset[0]+x.start+1, self._offset[0]+x.start+self._underlying[0],
            self._offset[1]+y.start+1, self._offset[1]+y.start+self._underlying[1],
            self._offset[2]+z.start+1, self._offset[2]+z.start+self._underlying[2])

        blob = self._bucket.get_blob(path)
        arr =  None
        if not blob:
            print ('could not find {}'.format(path))
            arr =  np.zeros(shape=(self._underlying[::-1]), dtype=np.uint8)
        else:
            seeked = blosc.decompress(blob.download_as_string()[9:])
            arr = np.fromstring(seeked, dtype=np.uint8).reshape(*self._underlying[::-1])
        return np.swapaxes(arr,0,2)
            
if __name__ == '__main__':
    Pinky()[0:2048,0:2048,0:64]

    16385-16448
    16449-16512
    16513-16576