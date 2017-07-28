# -*- coding: utf-8 -*-
import struct
import math
import triangle
import numpy as np
import json


class GlTF(object):

    def __init__(self):
        self.header = {}
        self.body = None

    def to_array(self):  # bgl
        scene = json.dumps(self.header, separators=(',', ':'))

        # scene = struct.pack(str(len(scene)) + 's', scene.encode('utf8'))
        # body must be 4-byte aligned
        scene += ' '*(4 - len(scene) % 4)

        binaryHeader = np.fromstring("glTF", dtype=np.uint8)
        binaryHeader2 = np.array([1,
                                  20 + len(self.body) + len(scene),
                                  len(scene),
                                  0], dtype=np.uint32)

        return np.concatenate((binaryHeader,
                               binaryHeader2.view(np.uint8),
                               np.fromstring(scene, dtype=np.uint8),
                               self.body))

    @staticmethod
    def from_array(array):
        """
        Parameters
        ----------
        array : numpy.array

        Returns
        -------
        glTF : GlTf
        """

        glTF = GlTF()

        if struct.unpack("4s", array[0:4])[0] != b"glTF":
            raise RuntimeError("Array does not contain a binary glTF")

        if struct.unpack("i", array[4:8])[0] != 1:
            raise RuntimeError("Unsupported glTF version")

        length = struct.unpack("i", array[8:12])[0]
        content_length = struct.unpack("i", array[12:16])[0]

        if struct.unpack("i", array[16:20])[0] != 0:
            raise RuntimeError("Unsupported binary glTF content type")

        header = struct.unpack(str(content_length) + "s",
                               array[20:20+content_length])[0]
        glTF.header = json.loads(header.decode("ascii"))
        glTF.body = array[20+content_length:length]

        return glTF

    @staticmethod
    def from_binary_arrays(arrays, transform, binary=True, batched=True, uri=None):
        """
        Parameters
        ----------
        arrays : array of dictionaries
            Each dictionary has the data for one geometry
            arrays['position']: binary array of vertex positions
            arrays['normal']: binary array of vertex normals
            arrays['uv']: binary array of vertex texture coordinates
            arrays['bbox']: geometry bounding box (numpy.array)

        transform : numpy.array
            World coordinates transformation flattend matrix

        Returns
        -------
        glTF : GlTF
        """

        glTF = GlTF()

        textured = 'uv' in arrays[0]

        binVertice = []
        binNormals = []
        binIds = []
        binUvs = []
        nVertice = []
        bb = []
        for i, geometry in enumerate(arrays):
            binVertice.append(geometry['position'])
            binNormals.append(geometry['normal'])
            n = round(len(geometry['position']) / 12)
            nVertice.append(n)
            bb.append(geometry['bbox'])
            if batched:
                binIds.append(np.full(n, i, dtype=np.uint16))
            if textured:
                binUvs.append(geometry['uv'])

        glTF.header = compute_header(binVertice, binNormals, binIds, binUvs,
                                     nVertice, bb, transform,
                                     binary, uri)
        glTF.body = np.frombuffer(
            compute_binary(binVertice, binNormals, binIds, binUvs), dtype=np.uint8)

        return glTF


def compute_binary(binVertices, binNormals, binIds, binUvs):
    bv = b''.join(binVertices)
    bn = b''.join(binNormals)
    bid = b''.join(binIds)
    buv = b''.join(binUvs)
    return bv + bn + bid + buv


def compute_header(binVertices, binNormals, binIds, binUvs,
                   nVertices, bb, transform,
                   bgltf, uri):
    batched = len(binIds) != 0
    textured = len(binUvs) != 0
    # Buffer
    meshNb = len(binVertices)
    sizeVce = []
    for i in range(0, meshNb):
        sizeVce.append(len(binVertices[i]))

    byteLength = 2 * sum(sizeVce)
    if batched:
        byteLength += 1 / 6 * sum(sizeVce)
    if textured:
        byteLength += sum(sizeVce)
    buffers = {
        'binary_glTF': {
            'byteLength': round(byteLength),
            'type': "arraybuffer"
        }
    }
    if uri is not None:
        buffers["binary_glTF"]["uri"] = uri

    # Buffer view
    bufferViews = {
        'BV_vertices': {
            'buffer': "binary_glTF",
            'byteLength': sum(sizeVce),
            'byteOffset': 0,
            'target': 34962
        },
        'BV_normals': {
            'buffer': "binary_glTF",
            'byteLength': sum(sizeVce),
            'byteOffset': sum(sizeVce),
            'target': 34962
        }
    }
    if textured:
        bufferViews['BV_uvs'] = {
            'buffer': "binary_glTF",
            'byteLength': round(2 / 3 * sum(sizeVce)),
            'byteOffset': 2 * sum(sizeVce),
            'target': 34962
        }
    if batched:
        bufferViews['BV_ids'] = {
            'buffer': "binary_glTF",
            'byteLength': sum(sizeVce) / 6,
            'byteOffset': round(8 / 9 * sum(sizeVce)) if textured else 2 * sum(sizeVce),
            'target': 34962
        }

    # Accessor
    accessors = {}
    if batched:
        accessors["AV"] = {
            'bufferView': "BV_vertices",
            'byteOffset': 0,
            'byteStride': 12,
            'componentType': 5126,
            'count': sum(nVertices),
            'max': [max([bb[i][0][1] for i in range(0, meshNb)]),
                    max([bb[i][0][2] for i in range(0, meshNb)]),
                    max([bb[i][0][0] for i in range(0, meshNb)])],
            'min': [min([bb[i][1][1] for i in range(0, meshNb)]),
                    min([bb[i][1][2] for i in range(0, meshNb)]),
                    min([bb[i][1][0] for i in range(0, meshNb)])],
            'type': "VEC3"
        }
        accessors["AN"] = {
            'bufferView': "BV_normals",
            'byteOffset': 0,
            'byteStride': 12,
            'componentType': 5126,
            'count': sum(nVertices),
            'max': [1, 1, 1],
            'min': [-1, -1, -1],
            'type': "VEC3"
        }
        accessors["AD"] = {
            'bufferView': "BV_ids",
            'byteOffset': 0,
            'byteStride': 2,
            'componentType': 5123,
            'count': sum(nVertices),
            'type': "SCALAR"
        }
        if textured:
            accessors["AT"] = {
                'bufferView': "BV_uvs",
                'byteOffset': 0,
                'byteStride': 8,
                'componentType': 5126,
                'count': sum(nVertices),
                'type': "VEC3"
            }
    else:
        for i in range(0, meshNb):
            accessors["AV_" + str(i)] = {
                'bufferView': "BV_vertices",
                'byteOffset': sum(sizeVce[0:i]),
                'byteStride': 12,
                'componentType': 5126,
                'count': nVertices[i],
                'max': [bb[i][0][1], bb[i][0][2], bb[i][0][0]],
                'min': [bb[i][1][1], bb[i][1][2], bb[i][1][0]],
                'type': "VEC3"
            }
            accessors["AN_" + str(i)] = {
                'bufferView': "BV_normals",
                'byteOffset': sum(sizeVce[0:i]),
                'byteStride': 12,
                'componentType': 5126,
                'count': nVertices[i],
                'max': [1, 1, 1],
                'min': [-1, -1, -1],
                'type': "VEC3"
            }
            if textured:
                accessors["AT_" + str(i)] = {
                    'bufferView': "BV_uvs",
                    'byteOffset': sum(sizeVce[0:i]),
                    'byteStride': 8,
                    'componentType': 5126,
                    'count': sum(nVertices),
                    'type': "VEC3"
                }

    # Meshes
    meshes = {}
    if batched:
        meshes["M"] = {
            'primitives': [{
                'attributes': {
                    "POSITION": "AV",
                    "NORMAL": "AN",
                    "BATCHID": "AD"
                },
                "material": "defaultMaterial",
                "mode": 4
            }]
        }
        if textured:
            meshes["M"]["primitives"][0]["attributes"]["TEXCOORD_0"] = "AT"
    else:
        for i in range(0, meshNb):
            meshes["M" + str(i)] = {
                'primitives': [{
                    'attributes': {
                        "POSITION": "AV_" + str(i),
                        "NORMAL": "AN_" + str(i)
                    },
                    "indices": "AI_" + str(i),
                    "material": "defaultMaterial",
                    "mode": 4
                }]
            }
            if textured:
                meshes["M" + str(i) ]["primitives"][0]["attributes"]["TEXCOORD_0"] = "AT_" + str(i)

    # Materials WIP
    if textured:
        meshes["M"]["primitives"][0]["material"] = 'textureMaterial'
            material = {
                'textureMaterial': {
                    'technique': "technique0",
                    "values": {
                        "diffuse": "atlas",
                }
                'name': "material0"
            }
    else:
        material = {
            'defaultMaterial': {
                'name': "None"
            }
        }
    }


    # Nodes
    if batched:
        nodes = {
            'node': {
                'matrix': [float(e) for e in transform],
                'meshes': ["M"]
            }
        }
    else:
        nodes = {
            'node': {
                'matrix': [float(e) for e in transform],
                'meshes': ["M" + str(i) for i in range(0, meshNb)]
            }
        }
    # TODO: one node per feature would probably be better

    # Extensions
    extensions = []
    if bgltf:
        extensions.append("KHR_binary_glTF")

    # Final glTF
    header = {
        'scene': "defaultScene",
        'scenes': {
            'defaultScene': {
                'nodes': [
                    "node"
                ]
            }
        },
        'nodes': nodes,
        'meshes': meshes,
        'accessors': accessors,
        'bufferViews': bufferViews,
        'buffers': buffers,
        'materials': material
        }
    }

    if textured:
        header["textures"] = {
            "texture_file2": {
                "format": 6408,
                "internalFormat": 6408,
                "sampler": "sampler_0",
                "source": "file2",
                "target": 3553,
                "type": 5121
            }
        }
        header["shaders"] = {
            "FS": {
                "type": 35632,
                "uri": "data:text/plain;base64,cHJlY2lzaW9uIGhpZ2hwIGZsb2F0Owp2YXJ5aW5nIHZlYzIgdl90ZXhjb29yZDA7CnVuaWZvcm0gc2FtcGxlcjJEIHVfZGlmZnVzZTsKdW5pZm9ybSB2ZWM0IHVfZW1pc3Npb247CnZvaWQgbWFpbih2b2lkKSB7CnZlYzQgY29sb3IgPSB2ZWM0KDAuLCAwLiwgMC4sIDAuKTsKdmVjNCBkaWZmdXNlID0gdmVjNCgwLiwgMC4sIDAuLCAxLik7CnZlYzQgZW1pc3Npb247CmRpZmZ1c2UgPSB0ZXh0dXJlMkQodV9kaWZmdXNlLCB2X3RleGNvb3JkMCk7CmVtaXNzaW9uID0gdV9lbWlzc2lvbjsKY29sb3IueHl6ICs9IGRpZmZ1c2UueHl6Owpjb2xvci54eXogKz0gZW1pc3Npb24ueHl6Owpjb2xvciA9IHZlYzQoY29sb3IucmdiICogZGlmZnVzZS5hLCBkaWZmdXNlLmEpOwpnbF9GcmFnQ29sb3IgPSBjb2xvcjsKfQo="
            },
            "VS": {
                "type": 35633,
                "uri": "data:text/plain;base64,cHJlY2lzaW9uIGhpZ2hwIGZsb2F0OwphdHRyaWJ1dGUgdmVjMyBhX3Bvc2l0aW9uOwp1bmlmb3JtIG1hdDQgdV9tb2RlbFZpZXdNYXRyaXg7CnVuaWZvcm0gbWF0NCB1X3Byb2plY3Rpb25NYXRyaXg7CmF0dHJpYnV0ZSB2ZWMyIGFfdGV4Y29vcmQwOwp2YXJ5aW5nIHZlYzIgdl90ZXhjb29yZDA7CnZvaWQgbWFpbih2b2lkKSB7CnZlYzQgcG9zID0gdV9tb2RlbFZpZXdNYXRyaXggKiB2ZWM0KGFfcG9zaXRpb24sMS4wKTsKdl90ZXhjb29yZDAgPSBhX3RleGNvb3JkMDsKZ2xfUG9zaXRpb24gPSB1X3Byb2plY3Rpb25NYXRyaXggKiBwb3M7Cn0K"
            }
        }
        header["samplers"] = {
            "sampler_0": {
                "magFilter": 9729,
                "minFilter": 9729,
                "wrapS": 10497,
                "wrapT": 10497
            }
        }

    # Technique for batched glTF
    """if batched:
        header['materials']['defaultMaterial']['technique'] = "T0"
        header['techniques'] = {
            "T0": {
                "attributes": {
                    "a_batchId": "batchId"
                },
                "parameters": {
                    "batchId": {
                        "semantic": "_BATCHID",
                        "type": 5123
                    }
                }
            }
        }
    """

    if len(extensions) != 0:
        header["extensionsUsed"] = extensions

    return header
