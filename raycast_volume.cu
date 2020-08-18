#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include <glm/glm.hpp>
#include <glm/gtx/component_wise.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>

// Compile with
// nvcc -Xcompiler -fPIC -shared -o raycast_volume.so raycast_volume.cu

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
extern "C"
__host__ __device__
void printVec(const char *str, glm::vec3 vec) {
  printf("%s: Vec3(%.2f, %.2f, %.2f)\n", str, vec.x, vec.y, vec.z);
}

__host__ __device__ float3 vec2float3(glm::vec3 v) {
  return make_float3(v.x, v.y, v.z);
}

texture<float, cudaTextureType3D, cudaReadModeElementType> tex;
texture<float, cudaTextureType1D, cudaReadModeElementType> tf;

__device__ inline float texFetchVolume(glm::vec3 p, bool useTF=false) {
  if (useTF) return tex1D(tf, tex3D(tex, p.x, p.y, p.z));
  else       return           tex3D(tex, p.x, p.y, p.z);
}

__device__ inline float texFetchVolumeWorld(glm::vec3 worldPos,
                                            glm::vec3 worldBounds, bool useTF=false) {
  glm::vec3 idx = worldPos / worldBounds;
  if (useTF) return tex1D(tf, tex3D(tex, idx.x, idx.y, idx.z));
  else       return           tex3D(tex, idx.x, idx.y, idx.z);
}

__device__ inline float texFetchVolumeWorld(float x, float y, float z, glm::vec3 worldBounds, bool useTF=false) {
  if (useTF) return tex1D(tf, tex3D(tex, x / worldBounds.x, y / worldBounds.y, z / worldBounds.z));
  else       return           tex3D(tex, x / worldBounds.x, y / worldBounds.y, z / worldBounds.z);
}

__device__
bool hasVisibleNeighbors(glm::vec3 pos, glm::vec3 volSize, bool useTF=true) {
    glm::vec3 offset = 1.f / volSize;
    float transparencySum = (
        texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, -1.f), useTF) +
        texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, 0.f),  useTF) +
        texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, 1.f),  useTF) +
        texFetchVolume(pos + offset * glm::vec3(-1.f, 0.f, -1.f),  useTF) +
        texFetchVolume(pos + offset * glm::vec3(-1.f, 0.f, 0.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(-1.f, 0.f, 1.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, -1.f),  useTF) +
        texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, 0.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, 1.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(0.f, -1.f, -1.f),  useTF) +
        texFetchVolume(pos + offset * glm::vec3(0.f, -1.f, 0.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(0.f, -1.f, 1.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(0.f, 0.f, -1.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(0.f, 0.f, 0.f),    useTF) +
        texFetchVolume(pos + offset * glm::vec3(0.f, 0.f, 1.f),    useTF) +
        texFetchVolume(pos + offset * glm::vec3(0.f, 1.f, -1.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(0.f, 1.f, 0.f),    useTF) +
        texFetchVolume(pos + offset * glm::vec3(0.f, 1.f, 1.f),    useTF) +
        texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, -1.f),  useTF) +
        texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, 0.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, 1.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(1.f, 0.f, -1.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(1.f, 0.f, 0.f),    useTF) +
        texFetchVolume(pos + offset * glm::vec3(1.f, 0.f, 1.f),    useTF) +
        texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, -1.f),   useTF) +
        texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, 0.f),    useTF) +
        texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, 1.f),    useTF)
    );
    return transparencySum > 0.f;
}

__device__ glm::vec3 computeGradientSobel(glm::vec3 pos, glm::ivec3 volSize,
                                          glm::vec3 voxelScale) {
  glm::vec3 offset = 0.1f / (glm::vec3(volSize) * voxelScale);
  float gx = (texFetchVolume(pos + offset * glm::vec3(1.f, 0.f, 0.f)) * 4.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, 0.f, 1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, 0.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, 0.f, -1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, 0.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, -1.f)) +
              texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, -1.f))) -
             (texFetchVolume(pos + offset * glm::vec3(-1.f, 0.f, 0.f)) * 4.f +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 0.f, 1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, 0.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 0.f, -1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, 0.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, -1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, -1.f)));

  float gy = (texFetchVolume(pos + offset * glm::vec3(0.f, 1.f, 0.f)) * 4.f +
              texFetchVolume(pos + offset * glm::vec3(0.f, 1.f, 1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, 0.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(0.f, 1.f, -1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, 0.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, -1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, -1.f))) -
             (texFetchVolume(pos + offset * glm::vec3(0.f, -1.f, 0.f)) * 4.f +
              texFetchVolume(pos + offset * glm::vec3(0.f, -1.f, 1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, 0.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(0.f, -1.f, -1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, 0.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, -1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, -1.f)));

  float gz = (texFetchVolume(pos + offset * glm::vec3(0.f, 0.f, 1.f)) * 4.f +
              texFetchVolume(pos + offset * glm::vec3(0.f, 1.f, 1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, 0.f, 1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(0.f, -1.f, 1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 0.f, 1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, 1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, 1.f))) -
             (texFetchVolume(pos + offset * glm::vec3(0.f, 0.f, -1.f)) * 4.f +
              texFetchVolume(pos + offset * glm::vec3(0.f, 1.f, -1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, 0.f, -1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(0.f, -1.f, -1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 0.f, -1.f)) * 2.f +
              texFetchVolume(pos + offset * glm::vec3(1.f, 1.f, -1.f)) +
              texFetchVolume(pos + offset * glm::vec3(1.f, -1.f, -1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, 1.f, -1.f)) +
              texFetchVolume(pos + offset * glm::vec3(-1.f, -1.f, -1.f)));

  return glm::vec3(gx, gy, gz);
}

__device__ glm::vec3 computeGradient(glm::vec3 worldPos, glm::vec3 worldBounds) {
  // glm::vec3 offset = 1.0f;// / worldBounds;
  float gx =
      texFetchVolumeWorld(worldPos.x + 1.0f, worldPos.y, worldPos.z, worldBounds) -
      texFetchVolumeWorld(worldPos.x - 1.0f, worldPos.y, worldPos.z, worldBounds);
  float gy =
      texFetchVolumeWorld(worldPos.x, worldPos.y + 1.0f, worldPos.z, worldBounds) -
      texFetchVolumeWorld(worldPos.x, worldPos.y - 1.0f, worldPos.z, worldBounds);
  float gz =
      texFetchVolumeWorld(worldPos.x, worldPos.y, worldPos.z + 1.0f, worldBounds) -
      texFetchVolumeWorld(worldPos.x, worldPos.y, worldPos.z - 1.0f, worldBounds);
  return glm::vec3(gx, gy, gz);
}

extern "C" cudaArray *uploadVolume(float *vol, glm::ivec3 volSizeVec) {
  cudaExtent volSize =
      make_cudaExtent(volSizeVec.x, volSizeVec.y, volSizeVec.z);
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  cudaArray *cuArray;
  gpuErrchk(cudaMalloc3DArray(&cuArray, &channelDesc, volSize));
  cudaMemcpy3DParms cpyParams = {0};
  cpyParams.srcPtr =
      make_cudaPitchedPtr((void *)vol, volSize.width * sizeof(float),
                          volSize.width, volSize.height);
  cpyParams.dstArray = cuArray;
  cpyParams.extent = volSize;
  cpyParams.kind = cudaMemcpyHostToDevice;
  gpuErrchk(cudaMemcpy3D(&cpyParams));

  tex.normalized = true;
  tex.filterMode = cudaFilterModeLinear;
  tex.addressMode[0] = cudaAddressModeBorder;
  tex.addressMode[1] = cudaAddressModeBorder;
  tex.addressMode[2] = cudaAddressModeBorder;
  gpuErrchk(cudaBindTextureToArray(tex, cuArray, channelDesc));

  return cuArray;
}

extern "C" cudaArray *uploadTransferFunction(float *tf_p,
                                             size_t tf_resolution) {
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  // float* d_tf;
  cudaArray *d_tf;
  // size_t offset = 0;
  // gpuErrchk(cudaMalloc((void**) &d_tf, tf_resolution * sizeof(float)));
  gpuErrchk(cudaMallocArray(&d_tf, &channelDesc, tf_resolution));
  // gpuErrchk(cudaMemcpy(d_tf, tf_p, sizeof(float) * tf_resolution,
  // cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy2DToArray(d_tf, 0, 0, tf_p, tf_resolution * sizeof(float),
                                tf_resolution * sizeof(float), 1,
                                cudaMemcpyHostToDevice));

  tf.normalized = true;
  tf.filterMode = cudaFilterModeLinear;
  tf.addressMode[0] = cudaAddressModeBorder;
  tf.addressMode[1] = cudaAddressModeBorder;
  // gpuErrchk(cudaBindTexture(&offset, tf, d_tf, channelDesc, sizeof(float) *
  // tf_resolution));
  gpuErrchk(cudaBindTextureToArray(tf, d_tf, channelDesc));

  return d_tf;
}

__device__
float accumulateOverRay(glm::vec3 rayStart, glm::vec3 step, int nSteps, glm::vec3 worldBounds, bool log=false) {
    float occlusion = 0.0f;
    glm::vec3 pos = rayStart + 3.f*step; // small offset
    for(int i = 0;
        i < nSteps &&           // Stop after max nSteps
        occlusion < 0.99f &&    // Early ray termination
        glm::all(glm::lessThan(pos,worldBounds)) &&           // out of volume
        glm::all(glm::greaterThan(pos,glm::vec3(0.f))); ++i){ // out of volume

      float op = texFetchVolumeWorld(pos, worldBounds, true);
      occlusion += (1-occlusion) * (op);
      // occlusion = max(occlusion, op);
      // if (occlusion > 0.5f) return 1.0f - static_cast<float>(i)/static_cast<float>(nSteps);
      pos += step;
      if (log && (op > 0.f || i > nSteps-5)) {
        printVec("Position", pos);
        printVec("World Bounds", worldBounds);
        printVec("Step", step);
        printVec("Ray Direction", glm::normalize(step));
        printf("Step %03d with Opacity=%1.3f\n\n==============\n", i, occlusion);
        }
    }
    return occlusion;
}

__device__ float sampleOpacityInRayDir(glm::vec3 vox_pos, glm::vec3 offset,
                                       glm::vec3 worldBounds){
    return (1.f * tex1D(tf, texFetchVolumeWorld(vox_pos + 1.f * offset, worldBounds)) +
            0.5f* tex1D(tf, texFetchVolumeWorld(vox_pos + 2.f * offset, worldBounds)) +
            0.33f*tex1D(tf, texFetchVolumeWorld(vox_pos + 3.f * offset, worldBounds)) +
            0.25f*tex1D(tf, texFetchVolumeWorld(vox_pos + 4.f * offset, worldBounds)) +
            0.125f*tex1D(tf, texFetchVolumeWorld(vox_pos + 8.f * offset, worldBounds))) / 2.205f;

}

__global__ void raycast(float *out, glm::ivec3 volSize, glm::vec3 rayDir,
                        float stepSize, glm::vec3 voxelScale, int nSteps) {
  // Get Volume index for this thread
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int i = 4 * (z * volSize.x * volSize.y + y * volSize.x + x);
  int nVoxel = glm::compMul(volSize);
  if (i >= 4 * nVoxel) {
    printf("Out of bounds with threading idxs");
    return;
  }
  glm::vec3 worldPos = (glm::vec3(x, y, z) + 0.5f) * voxelScale;
  glm::vec3 worldBounds = glm::vec3(volSize) * voxelScale;

  glm::vec3 normal = glm::normalize(-computeGradient(worldPos, worldBounds));
  if (glm::any(glm::isnan(normal))){ normal = glm::vec3(0.f); }
  float weight = glm::clamp(glm::dot(rayDir, normal), 0.f, 1.f);
  float intensity = texFetchVolumeWorld(worldPos, worldBounds);
  float tfd = tex1D(tf, intensity);
  out[i + 0] = normal.x;
  out[i + 1] = intensity;
  out[i + 2] = tfd;

  out[i + 3] += 1.f - accumulateOverRay(worldPos, stepSize * rayDir, nSteps, worldBounds, false);
  // if (weight > 0.f && hasVisibleNeighbors(worldPos / worldBounds, volSize, true)) {
  //   float opacity = accumulateOverRay(worldPos, stepSize * rayDir, nSteps, worldBounds, false);
  //   out[i + 3] += weight * (1 - opacity);
  // }
}

// __device__
// float median(std::vector<float> &v) {
//   size_t idx = v.size() / 2;
//   std::nth_element(v.begin(), v.begin() + idx, v.end());
//   return v[idx];
// }

__device__ float medianBubble(float *v, size_t sz) {
  size_t minValueIndex;
  float bufferData;
  size_t i, j;

  for (j = 0; j <= (sz - 1) / 2; j++) {
    minValueIndex = j;
    for (i = j + 1; i < sz; i++)
      if (v[i] < v[minValueIndex])
        minValueIndex = i;

    bufferData = v[j];
    v[j] = v[minValueIndex];
    v[minValueIndex] = bufferData;
  }

  return v[(sz - 1) / 2];
}

__global__ void replaceWithMedian(float* vol, glm::ivec3 volSize, float value, float *out) {
  size_t X = volSize.x, Y = volSize.y, Z = volSize.z;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int i = z * X * Y + y * X + x;
  if(vol[i] != value) return;
  glm::ivec3 shifts[26];
  shifts[0]  = glm::ivec3(0, 0, 1);
  shifts[1]  = glm::ivec3(0, 0, -1);
  shifts[2]  = glm::ivec3(0, 1, 0);
  shifts[3]  = glm::ivec3(0, -1, 0);
  shifts[4]  = glm::ivec3(1, 0, 0);
  shifts[5]  = glm::ivec3(-1, 0, 0);
  shifts[6]  = glm::ivec3(0, 1, 1);
  shifts[7]  = glm::ivec3(0, 1, -1);
  shifts[8]  = glm::ivec3(0, -1, 1);
  shifts[9]  = glm::ivec3(0, -1, -1);
  shifts[10] = glm::ivec3(1, 0, 1);
  shifts[11] = glm::ivec3(1, 0, -1);
  shifts[12] = glm::ivec3(-1, 0, 1);
  shifts[13] = glm::ivec3(-1, 0, -1);
  shifts[14] = glm::ivec3(1, 1, 0);
  shifts[15] = glm::ivec3(1, -1, 0);
  shifts[16] = glm::ivec3(-1, 1, 0);
  shifts[17] = glm::ivec3(-1, -1, 0);
  shifts[18] = glm::ivec3(1, 1, 1);
  shifts[19] = glm::ivec3(1, 1, -1);
  shifts[20] = glm::ivec3(1, -1, 1);
  shifts[21] = glm::ivec3(1, -1, -1);
  shifts[22] = glm::ivec3(-1, 1, 1);
  shifts[23] = glm::ivec3(-1, 1, -1);
  shifts[24] = glm::ivec3(-1, -1, 1);
  shifts[25] = glm::ivec3(-1, -1, -1);

  float vec[26];
  size_t count = 0;
  for (size_t i=0; i < 26; ++i) {
    if (0 < x + shifts[i].x && x + shifts[i].x < X &&
        0 < y + shifts[i].y && y + shifts[i].y < Y &&
        0 < z + shifts[i].z && z + shifts[i].z < Z) {
      size_t cur_i = (z + shifts[i].z)*X*Y  +  (y + shifts[i].y)*X  +  (x + shifts[i].x);
      vec[count] = vol[cur_i];
      count++;
    }
  }
  out[i] = medianBubble(vec, count);
}

extern "C"
void raycastVolume(float* vol_p, float* tf_p, int* texDims_p, float* voxelScale_p, float* ray_p, float stepsFactor, size_t nRays, float minValue, float* vol_out_p){
    // printf("raycastVolume()\n");
    glm::ivec3 volSize      = glm::ivec3(texDims_p[0], texDims_p[1], texDims_p[2]);
    size_t     tfResolution = texDims_p[3];
    glm::vec3  voxelScale   = glm::make_vec3(voxelScale_p);
    size_t     nVoxel       = static_cast<size_t>(glm::compMul(volSize));
    float      stepSize     = 0.5f;
    size_t     nSteps       = static_cast<size_t>(ceil(glm::length(glm::vec3(volSize) * voxelScale) * stepsFactor / stepSize));
    // Metadata
    printf("Raycasting Volume (%d, %d, %d, Total: %d). TF has res of %d.\n", volSize.x, volSize.y, volSize.z, nVoxel, tfResolution);
    printf("Doing %d steps of length %1.4f for a total ray length of %1.2f\n", nSteps, stepSize, nSteps * stepSize);


    // float4* vol_out_f4p = (float4 *) vol_out_p;
    // Allocate result memory on gpu
    float* d_vol_out;
    gpuErrchk(cudaMalloc((void **) &d_vol_out, nVoxel * 4 * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_vol_out, vol_out_p, nVoxel * 4 * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid(    1,     volSize.y, volSize.z);
    dim3 block(volSize.x,    1,          1   );
    // Replace invalid voxels (==vol.min()) with median of neighborhood
    // float *d_vol, *d_vol_r;
    // gpuErrchk(cudaMalloc((void **) &d_vol,   nVoxel * sizeof(float)));
    // gpuErrchk(cudaMalloc((void **) &d_vol_r, nVoxel * sizeof(float)));
    // gpuErrchk(cudaMemcpy(d_vol, vol_p,       nVoxel * sizeof(float), cudaMemcpyHostToDevice));
    // replaceWithMedian<<<grid, block>>>(d_vol, volSize, minValue, d_vol_r);
    // gpuErrchk(cudaDeviceSynchronize());
    // gpuErrchk(cudaMemcpy(vol_p, d_vol_r, nVoxel * sizeof(float), cudaMemcpyDeviceToHost));
    // Upload volume as texture
    auto d_vol_tex = uploadVolume(vol_p, volSize);
    auto d_tf_tex = uploadTransferFunction(tf_p, tfResolution);
    gpuErrchk(cudaPeekAtLastError());

    gpuErrchk(cudaPeekAtLastError());
    // printf("Starting now...\n");
    for (size_t i = 0; i < 3*nRays; i+=3){
      glm::vec3 rayDir = glm::vec3(ray_p[i], ray_p[i + 1], ray_p[i + 2]);
      // glm::vec3 rayDir = glm::normalize(glm::vec3(ray_p[i], ray_p[i + 1], ray_p[i + 2]) / voxelScale);
      raycast<<<grid, block>>>(d_vol_out, volSize, rayDir, stepSize, voxelScale, nSteps);
      // if(rayDir.z > 0.f) { printf("!!!!!!!!!!! we got z > 0"); printVec("rayDir", rayDir); }
      gpuErrchk( cudaDeviceSynchronize() );
      // printf("Rays cast: %d/%d (%02.2f%)\r", i/3, nRays, 100.f*(i/3.0f)/nRays);
      // std::cout.flush();
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(vol_out_p, d_vol_out, nVoxel * 4 * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk( cudaFree(d_vol_out));
    gpuErrchk( cudaFreeArray(d_tf_tex));
    gpuErrchk( cudaFreeArray(d_vol_tex));
    gpuErrchk( cudaUnbindTexture(tex));
    gpuErrchk( cudaUnbindTexture(tf));
}
