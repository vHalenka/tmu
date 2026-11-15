// Encode a single patch size or several sizes directly from GPU pixel data.
// All sizes share one literal layout; each size writes its own patch range.
#include <stddef.h>
extern "C" __global__ void tc_encode_patch_scale(
    const unsigned *raw, unsigned *packed, const unsigned char *orientation,
    int n,int w,int h,int d,int pw,int ph,int features,int chunks,
    int mw,int mh,int nx,int ny,int scale,int scales,int total_patches,
    int offset,int multi,int append_negated,int reflect_patch)
{
    const size_t patches=(size_t)(w-pw+1)*(h-ph+1);
    const size_t words=(size_t)n*patches*chunks;
    const size_t pixels=(size_t)mw*mh;
    for(size_t word=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
        word<words;word+=(size_t)gridDim.x*blockDim.x){
        const size_t image=word/(patches*chunks),position=(word/chunks)%patches;
        const int y=position/(w-pw+1),x=position%(w-pw+1);
        unsigned result=0;
        for(int bit=0;bit<32;++bit){
            const size_t literal=(word%chunks)*32+bit;
            if(literal>=(size_t)features*(append_negated?2:1))break;
            const size_t feature=literal%features;
            unsigned value;
            if(feature<(size_t)ny)value=y>feature;
            else if(feature<(size_t)ny+nx)value=x>feature-ny;
            else if(feature<(size_t)ny+nx+pixels*d){
                const size_t k=feature-ny-nx;
                const int py=k/((size_t)mw*d),px=(k/d)%mw,z=k%d;
                const unsigned orient=reflect_patch ? orientation[image*patches+position] : 0;
                const int read_x=(orient & 1u) ? pw-1-px : px;
                const int read_y=(orient & 2u) ? ph-1-py : py;
                value=py<ph&&px<pw ? raw[((image*h+y+read_y)*w+x+read_x)*d+z]==1u : 0;
            }else if(multi&&feature<(size_t)ny+nx+pixels*(d+1)){
                const size_t k=feature-ny-nx-pixels*d;
                value=k/mw<(size_t)ph&&k%mw<(size_t)pw;
            }else value=feature-ny-nx-pixels*(d+1)==(size_t)scale;
            if(literal>=(size_t)features)value=!value;
            result|=value<<bit;
        }
        packed[((image*(size_t)total_patches+offset+position)*chunks)+(word%chunks)]=result;
    }
}

// Canonical whole images follow caller NHW/NHWC rows/columns. Channels stay
// together and only ==1 is true. Compare all allowed orientations as a group.
extern "C" __global__ void tc_canonical_images(
    const unsigned *raw,unsigned *out,unsigned char *flags,int n,int rows,int columns,int depth,int axes)
{
    __shared__ int orientation;
    const size_t pixels=(size_t)rows*columns*depth;
    for(size_t image=blockIdx.x;image<(size_t)n;image+=gridDim.x){
        const size_t base=image*pixels;
        if(threadIdx.x==0){
            orientation=0;
            for(int candidate=1;candidate<4;++candidate){
                if(candidate & ~axes)continue;
                for(size_t k=0;k<pixels;++k){
                    const size_t row=k/((size_t)columns*depth),column=(k/depth)%columns,z=k%depth;
                    const size_t ar=(orientation & 2) ? rows-1-row : row;
                    const size_t ac=(orientation & 1) ? columns-1-column : column;
                    const size_t br=(candidate & 2) ? rows-1-row : row;
                    const size_t bc=(candidate & 1) ? columns-1-column : column;
                    const unsigned a=raw[base+(ar*columns+ac)*depth+z]==1u;
                    const unsigned b=raw[base+(br*columns+bc)*depth+z]==1u;
                    if(a!=b){if(b<a)orientation=candidate;break;}
                }
            }
            flags[image]=(unsigned char)orientation;
        }
        __syncthreads();
        for(size_t k=threadIdx.x;k<pixels;k+=blockDim.x){
            const size_t row=k/((size_t)columns*depth),column=(k/depth)%columns,z=k%depth;
            const size_t r=(orientation & 2) ? rows-1-row : row;
            const size_t c=(orientation & 1) ? columns-1-column : column;
            out[base+k]=raw[base+(r*columns+c)*depth+z];
        }
        __syncthreads();
    }
}

// Local window orientation changes content only; the window stays in place.
extern "C" __global__ void tc_choose_patch_orientation(
    const unsigned *raw,unsigned char *flags,int n,int w,int h,int d,int pw,int ph,int axes)
{
    const size_t patches=(size_t)(w-pw+1)*(h-ph+1),total=(size_t)n*patches;
    for(size_t id=(size_t)blockIdx.x*blockDim.x+threadIdx.x;id<total;id+=(size_t)gridDim.x*blockDim.x){
        const size_t image=id/patches,position=id%patches;
        const int y=position/(w-pw+1),x=position%(w-pw+1);
        unsigned char orientation=0;
        for(int candidate=1;candidate<4;++candidate){
            if(candidate & ~axes)continue;
            for(size_t k=0;k<(size_t)pw*ph*d;++k){
                const int py=k/((size_t)pw*d),px=(k/d)%pw,z=k%d;
                const int ar=(orientation & 2) ? ph-1-py : py;
                const int ac=(orientation & 1) ? pw-1-px : px;
                const int br=(candidate & 2) ? ph-1-py : py;
                const int bc=(candidate & 1) ? pw-1-px : px;
                const unsigned a=raw[((image*h+y+ar)*w+x+ac)*d+z]==1u;
                const unsigned b=raw[((image*h+y+br)*w+x+bc)*d+z]==1u;
                if(a!=b){if(b<a)orientation=candidate;break;}
            }
        }
        flags[id]=orientation;
    }
}
