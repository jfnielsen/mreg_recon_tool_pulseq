
#include "mex.h"
#include "matrix.h"
#include <math.h>
#include <complex>
#include <unistd.h>
#include "fftw3.h"



#include "cufft.h"
#include "cuda_runtime.h"
#include <cuda.h> 
//#include <cublas.h>


#include <stdio.h>
#include <string>
#include <iostream>
#include <sys/time.h>


#include <string.h>
#include <sys/time.h>

#define GET_TIME(now) {				\
    struct timeval t;				\
    gettimeofday(&t, NULL);			\
    now = t.tv_sec + t.tv_usec/1000000.0;	\
  }


#define MAX_BLOCK_SZ 2048

#include "tikreg_CG_kernels_d_off.cu"

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
 	
  if(nrhs != 9 ) {
    printf("\nUsage:\n");
    return;
  } else if(nlhs>1) {
    printf("Too many output arguments\n");
    return;
  }

  //////////////////////////////////// fetching data from MATLAB

  int pcnt = 0;  
  const mxArray *Measurement;
  Measurement = prhs[pcnt++];       
  std::complex<double> *meas = ( std::complex<double> *) mxGetData(Measurement);
    

    
  const mxArray *Sens;
  Sens = prhs[pcnt++];       
  const int *dims_sens = mxGetDimensions(Sens);
  const int numdim_sens = mxGetNumberOfDimensions(Sens);
  std::complex<double> *sens = ( std::complex<double> *) mxGetData(Sens);
    
  int numsens;
  if (numdim_sens == 4)
    numsens = 1;
  else
    numsens = dims_sens[4];
        
  const int numdim =4;
  const int dims_sz[] = {2, dims_sens[1], dims_sens[2], dims_sens[3] };
  int w = (int)dims_sz[1];
  int h = (int)dims_sz[2];
  int d = (int)dims_sz[3];
  int totsz = w*h*d;
           
 
    //Modified by Sandy & Jussi (Begin)
    const mxArray *Ipk_index;
    Ipk_index = prhs[pcnt++];       
    const int numSeg = mxGetM(Ipk_index);
    const mwSize *dims_ipk[numSeg];
	int numP;
	int numK[numSeg];
	int *ipk_index[numSeg];
	mxArray *seg;
	for(j=0;j<numSeg;j++)
	{
		seg=mxGetCell(Ipk_index, j);
		dims_ipk[j]  = mxGetDimensions(seg);
		ipk_index[j] = (int*) mxGetData(seg);
  
		numK[j] = (dims_ipk[j])[1];
	}
	numP = (dims_ipk[1])[0];

    const mxArray *Ipk_we;
    Ipk_we = prhs[pcnt++];       
    std::complex<double> *ipk_we[numSeg];
	for(j=0;j<numSeg;j++)
	{
		seg=mxGetCell(Ipk_we, j);
		ipk_we[j] = (std::complex<double>*) mxGetData(seg);
	}

    const mxArray *Dims_pad;
    Dims_pad = prhs[pcnt++];       
    double *dims_pad_d = (double*) mxGetData(Dims_pad);
    int w_pad = (int)dims_pad_d[0];
    int h_pad = (int)dims_pad_d[1];
    int d_pad = (int)dims_pad_d[2];
    int totsz_pad  = w_pad*h_pad*d_pad;

    
    const mxArray *BPidx;
    BPidx = prhs[pcnt++]; 
	int *bpidx[numSeg];
	int numVox[numSeg];
	for(j=0;j<numSeg;j++)
	{
		seg=mxGetCell(BPidx, j);
		numVox[j]=mxGetM(seg);
		bpidx[j]=(int*) mxGetData(seg);
	}
    //int numVox= mxGetM(BPidx);
    //int * bpidx = (int*) mxGetData(BPidx);
    
    const mxArray *BPmidx;
    BPmidx = prhs[pcnt++];       
   
    const mxArray *BPweight;
    BPweight = prhs[pcnt++];       

	const mxArray *TrajLen;
	TrajLen = prhs[pcnt++];
	int numKT;
	numKT=((int*)mxGetData(TrajLen))[0];

	const mxArray *PMaps;
    PMaps = prhs[pcnt++];       
    std::complex<double> *pmap = ( std::complex<double> *) mxGetData(PMaps);

	const mxArray *Segidx;
	Segidx = prhs[pcnt++];       
	int *segidx;
	segidx=(int*) mxGetData(Segidx);

	const mxArray *SegFil;
	SegFil = prhs[pcnt++];       
	double *segfil[numSeg];
	for(j=0;j<numSeg;j++)
	{
		seg=mxGetCell(SegFil, j);
		segfil[j]=(double*) mxGetData(seg);
	}
    //Modified by Sandy & Jussi(End)

  const mxArray *Params;
  Params = prhs[pcnt++];       
  double *params = (double*) mxGetData(Params);
  int numit = (int) params[0];
  double lambda = params[1];
  int device_num = (int) params[2];
  double tol = params[3];
  int VERBOSE = (int) params[4];
    
  if (VERBOSE == 1)  
    mexPrintf("gpuDevice: %i  lambda^2: %f %d\n",device_num,lambda, MAX_BLOCK_SZ);

  /**************** Init Cuda *****************/
    
  cudaError_t rv; 
  CUdevice dev; 
    
  if (cuCtxGetDevice(&dev) == CUDA_SUCCESS)
    {
      //   CUcontext  pctx ;
      //   cuCtxPopCurrent(&pctx);	      
    }   
    
  mexPrintf("dev:%i\n",dev);
    
  //     
  //    cudaSetDevice(device_num); 
  // 
  //    rv = cudaSetDeviceFlags(cudaDeviceMapHost);
  //    if (rv != cudaSuccess )
  //    {
  //       mexPrintf("Call to cudaSetDeviceFlags failed\n");
  //       return;
  //    } 
  //     
  /******** Allocate mapped tmps for dot product calc **********/
   
  int dot_threads = 128;
  int dot_blocks = w*h*d/dot_threads;
   
  double *dot_z_h,*dot_z_d;
  rv = cudaHostAlloc(&dot_z_h, dot_blocks*sizeof(double), cudaHostAllocMapped);

  if (rv != cudaSuccess) {
    mexPrintf("Call to cudaHostAlloc failed: %i\n",rv);
    return;
  } 
  cudaHostGetDevicePointer(&dot_z_d, dot_z_h, 0);
  cudaMemset(dot_z_d,0,dot_blocks*sizeof(double));
   
   
  /////////////////////////////////////// MALLOCs
    
  double start,finish;
     
  GET_TIME(start);
    
  cufftDoubleComplex *tmp1,*tmp2,*tmp3, *_r, *_d, *_z , *_meas,*_sens, *_ipk_we, *tmpsens;
  int *_the_index;
  // added by jussi
  double *_segfil[numSeg];
  cufftHandle            plan;
    
  plhs[0]             =  mxCreateNumericArray(numdim,dims_sz,mxGetClassID(Sens),mxREAL);
     
  std::complex<double> *res = (std::complex<double> *) mxGetData(plhs[0]);
   
  cudaMalloc( (void **) &tmp1,sizeof(cufftDoubleComplex)*totsz_pad);
  cudaMalloc( (void **) &tmp2,sizeof(cufftDoubleComplex)*totsz_pad);
  cudaMalloc( (void **) &tmp3,sizeof(cufftDoubleComplex)*totsz);
  cudaMalloc( (void **) &_r,sizeof(cufftDoubleComplex)*totsz);
  cudaMalloc( (void **) &_d,sizeof(cufftDoubleComplex)*totsz);
  cudaMalloc( (void **) &_z,sizeof(cufftDoubleComplex)*totsz);
  //  cudaMalloc( (void **) &_meas,sizeof(cufftDoubleComplex)*numsens*numK);

    //Modified by Sandy (Begin)
    cudaMalloc( (void **) &_meas,sizeof(cufftDoubleComplex)*numsens*numKT);
    cudaMalloc( (void **) &_sens,sizeof(cufftDoubleComplex)*numsens*totsz);
    cudaMalloc( (void **) &_pmap,sizeof(cufftDoubleComplex)*numSeg*totsz);
    for(int j=0;j<numSeg;j++)
      {
	cudaMalloc((void **)&_segfil[j],sizeof(double)*numK[j]);
	cudaMalloc((void **)&_ipk_we[j],sizeof(cufftDoubleComplex)*numP*numK[j]);
	cudaMalloc((void **)&_the_index[j],sizeof(int)*numP*numK[j]);
      }
    cudaMalloc((void **)&tmpsens,sizeof(cufftDoubleComplex)*numKT);
    //Modified by Sandy (End)
    
  //  cudaMalloc( (void **) &_sens,sizeof(cufftDoubleComplex)*numsens*totsz);
  //  cudaMalloc( (void **) &_ipk_we,sizeof(cufftDoubleComplex)*numP*numK);
  //  cudaMalloc( (void **) &_the_index,sizeof(int)*numP*numK);
  //  cudaMalloc( (void **) &tmpsens,sizeof(cufftDoubleComplex)*numK);

  cudaThreadSynchronize();
   
  cudaMemset( tmp1,0,sizeof(cufftDoubleComplex)*totsz_pad);
  cudaMemset( tmp2,0,sizeof(cufftDoubleComplex)*totsz_pad);
  cudaMemset(  tmp3,0,sizeof(cufftDoubleComplex)*totsz);
  cudaMemset(  _r,0,sizeof(cufftDoubleComplex)*totsz);
  cudaMemset(  _d,0,sizeof(cufftDoubleComplex)*totsz);
  cudaMemset(  _z,0,sizeof(cufftDoubleComplex)*totsz);
 
  cudaThreadSynchronize();
 
  
  /************** copy data on device **********************/

  cudaMemcpy( _meas, meas, sizeof(cufftDoubleComplex)*numsens*numK, cudaMemcpyHostToDevice);
  cudaMemcpy( _sens, sens, sizeof(cufftDoubleComplex)*numsens*totsz, cudaMemcpyHostToDevice);
  //  cudaMemcpy( _ipk_we, ipk_we, sizeof(cufftDoubleComplex)*numP*numK, cudaMemcpyHostToDevice);
  //cudaMemcpy( _the_index, the_index, sizeof(int)*numP*numK, cudaMemcpyHostToDevice);
   
  //cudaMemcpy( ipk_we, _ipk_we, sizeof(cufftDoubleComplex)*numP*numK, cudaMemcpyDeviceToHost);
  //cudaMemcpy( the_index, _the_index, sizeof(int)*numP*numK, cudaMemcpyDeviceToHost);
     
     //Modified by Sandy (Begin)
    cudaMemcpy( _pmap, pmap,
			sizeof(cufftDoubleComplex)*numSeg*totsz, cudaMemcpyHostToDevice);

	for(j=0;j<numSeg;j++)
	{
		// what?  why?  jussi2020
		cudaMemcpy( _segfil[j], segfil[j],
				sizeof(double)*numK[j], cudaMemcpyHostToDevice);
		cudaMemcpy(_ipk_we[j], ipk_we[j],
				sizeof(cufftDoubleComplex)*numP*numK[j], cudaMemcpyHostToDevice);
		cudaMemcpy(_the_index[j], ipk_index[j],
				sizeof(int)*numP*numK[j], cudaMemcpyHostToDevice);
   
		cudaMemcpy(segfil[j], _segfil[j],
				sizeof(double)*numK[j], cudaMemcpyDeviceToHost);
		cudaMemcpy(ipk_we[j], _ipk_we[j], 
				sizeof(cufftDoubleComplex)*numP*numK[j], cudaMemcpyDeviceToHost);
		cudaMemcpy(ipk_index[j], _the_index[j],
				sizeof(int)*numP*numK[j], cudaMemcpyDeviceToHost);
	}
    //Modified by Sandy (End)

  cudaThreadSynchronize();
    
  if (VERBOSE == 1) 
    mexPrintf("numP: %i  numK: %i whd %i %i %i pad %i %i %i numsens: %i\n",numP,numK,w,h,d,w_pad,h_pad,d_pad,numsens);
            
      
    /************** copy bpidx on device **********************/
    //Modified by Sandy (Begin)
    int *_bpmidx[numSeg];
    cufftDoubleComplex *_bpweight[numSeg];
	int *_bpsize[numSeg], *_bponset[numSeg], *_bpidx[numSeg];

	for(j=0;j<numSeg;j++)
	{
	
		int *bpsize = (int*) malloc(sizeof(int)*numVox[j]);
		int *bponset  = (int*) malloc(sizeof(int)*(numVox[j]+1));
		bponset[0] = 0;
		mxArray *BP_midx=mxGetCell(BPmidx, j); 
		mxArray *BP_weight=mxGetCell(BPweight, j);

		for (i=0;i<numVox[j];i++)
		{
			mxArray *Midx = mxGetCell(BP_midx,i);
			bpsize[i] = mxGetM(Midx);
			bponset[i+1] = bponset[i] + bpsize[i];
		}
		int *tmp_bpmidx; 
		cufftDoubleComplex *tmp_bpweight;

		tmp_bpmidx = (int*) malloc(sizeof(int)*bponset[numVox[j]]);
		tmp_bpweight = (cufftDoubleComplex*) malloc(
					sizeof(cufftDoubleComplex)*bponset[numVox[j]]
					);
    
		if (tmp_bpmidx == 0)
		{
			mexPrintf("out of mem (host %d)\n", j);
			return;
		}
		if (tmp_bpweight == 0)
		{
			mexPrintf("out of mem (host %d)\n", j);
			return;
		}

		for (i=0;i<numVox[j];i++)
		{
			mxArray *Midx = mxGetCell(BP_midx,i);
			mxArray *Weight = mxGetCell(BP_weight,i);

			int *midx = (int*)  mxGetData(Midx);
			cufftDoubleComplex *bpwei = (cufftDoubleComplex*) mxGetData(Weight);
			memcpy(tmp_bpmidx + bponset[i] , midx, sizeof(int)* bpsize[i]);
			memcpy(tmp_bpweight + bponset[i] , 
				bpwei, sizeof(cufftDoubleComplex)* bpsize[i]);    
		}
      
		cudaMalloc( (void **) &_bpmidx[j],
				sizeof(int)* bponset[numVox[j]]);
		cudaMalloc( (void **) &_bpweight[j],
				sizeof(cufftDoubleComplex)* bponset[numVox[j]]);
      
		cudaMemcpy(_bpmidx[j],tmp_bpmidx,
				sizeof(int)*bponset[numVox[j]],
				cudaMemcpyHostToDevice);
		cudaMemcpy(_bpweight[j],tmp_bpweight,
				sizeof(cufftDoubleComplex)*bponset[numVox[j]], 
				cudaMemcpyHostToDevice);
    
		cudaMalloc( (void **) &_bpsize[j],sizeof(int)* numVox[j]);   
		cudaMalloc( (void **) &_bpidx[j],sizeof(int)* numVox[j]);
		cudaMalloc( (void **) &_bponset[j],sizeof(int)* numVox[j]+1);    

		cudaMemcpy(_bpidx[j],bpidx[j],
				sizeof(int)* numVox[j], cudaMemcpyHostToDevice);
		cudaMemcpy(_bpsize[j],bpsize,
				sizeof(int)* numVox[j], cudaMemcpyHostToDevice);
		cudaMemcpy(_bponset[j],bponset,
				sizeof(int)* numVox[j]+1, cudaMemcpyHostToDevice);

		free(tmp_bpmidx);
    	free(tmp_bpweight);
    	free(bpsize);
    	free(bponset);
	}
    cudaThreadSynchronize();
    //Modified by Sandy (End)
            
  GET_TIME(finish);
    
  if (VERBOSE == 1) {
    mexPrintf("num active Vox: %i\n",numVox);    
    mexPrintf("alloc/copy time: %f\n",finish-start);
  }
        
  cufftPlan3d(&plan, d_pad, h_pad, w_pad, CUFFT_Z2Z) ;
        
  
     
  // thread managements 
  int vx_block = 128;
  dim3 dimBlock_vx(vx_block,1);
  dim3 dimGrid_vx (numVox/vx_block + 1,1);
 
  dim3 dimBlock_dw(d,1);
  dim3 dimGrid_dw (w,h);

  dim3 dimBlock_sq(d,1);
  dim3 dimGrid_sq (w*h,1);
  
  // for sensing 
  int sens_block = 256;
  dim3 dimBlock_se(sens_block,1);
  dim3 dimGrid_se (numK/sens_block + 1,1);

	dim3 dimGrid_se[numSeg];
	for(j=0;j<numSeg;j++)
	{
		dimGrid_se[j]=dim3(numK[j]/sens_block + 1,1);
	}
 
    
  double AA_time = 0;
  double cg_time = 0;
     
  int err;
    
   
  double normrr = 0;
  double dAAd = 0;
  double alpha = 0;
  double normrr2 = 0;
  double beta = 0;
      
  /////////////////////////////////////////////////////// init CG
    

  // we need this because first fft fails
  int _res = cufftExecZ2Z(plan, tmp1, tmp2, CUFFT_FORWARD) ;
  if (VERBOSE == 1)
    mexPrintf("first fft call ret: %i\n",_res);

     
  cudaMemset(_r,0, sizeof(cufftDoubleComplex)*totsz);
  cudaMemset(tmp3,0, sizeof(cufftDoubleComplex)*totsz);
  cudaMemset(_z,0, sizeof(cufftDoubleComplex)*totsz);   
  cudaMemset( tmp2,0,sizeof(cufftDoubleComplex)*totsz_pad);
   
                
  // backproject measurement -- x=A'b
  for (int i = 0; i < numsens; i++)
    { 
      cudaMemset(tmp1,0, sizeof(cufftDoubleComplex)*totsz_pad);
      //        backprojVX<<<dimGrid_vx,dimBlock_vx>>>(_bpidx,_bponset,_bpweight,_bpmidx,_bpsize,_meas + i*numK, tmp1,numVox);
      backprojVX<<<dimGrid_vx,dimBlock_vx>>>(_bpidx,_bponset,_bpweight,_bpmidx,_bpsize,_meas + i*numK, _segfil[i], tmp1,numVox);
      if (err=cufftExecZ2Z(plan, tmp1, tmp2, CUFFT_INVERSE) != CUFFT_SUCCESS)
        {
	  mexPrintf("1) cufft has failed with err %i \n",err);
	  return;
        }    
      downwind<<<dimGrid_dw,dimBlock_dw>>>(_r,tmp2, _sens + i*totsz, w, h, d, w_pad, h_pad, d_pad);
    }
  
  cudaMemcpy( res, _r, sizeof(cufftDoubleComplex)*totsz,cudaMemcpyDeviceToHost);    
  cudaMemcpy( _d, _r, sizeof(cufftDoubleComplex)*totsz,cudaMemcpyDeviceToDevice);
     
    
  normrr = Dot_wrapper(_r, _r, dot_z_d, dot_z_h, totsz, dot_blocks, dot_threads) ;
     
  // normrr = cublasCdotc(totsz,(cuComplex*)_r,1,(cuComplex*)_r,1).x;           
     
  double normrr0 = normrr;
  if (VERBOSE == 1)
    mexPrintf("first residual: %f\n",normrr0);
   
    
     
  ////////////////////////////////////////////////////////////// start CG
  GET_TIME(start);
   
  for (int it = 0; it < numit; it++)
    {
     
      GET_TIME(start);
      scmult<<<dimGrid_sq,dimBlock_sq>>>(tmp3,_d,lambda,totsz);
      GET_TIME(finish); cg_time += finish-start;
        
      for (int i = 0; i < numsens; i++)
        {
            
	  cudaMemset(tmp1,0, sizeof(cufftDoubleComplex)*totsz_pad);           
	  upwind<<<dimGrid_dw,dimBlock_dw>>>(tmp1,_d, _sens + i*totsz, w, h, d, w_pad, h_pad, d_pad);            
	  if (err=cufftExecZ2Z(plan, tmp1, tmp2, CUFFT_FORWARD) != CUFFT_SUCCESS)
            {
	      mexPrintf("2) cufft has failed with err %i \n",err);
	      return;
            }
	  cudaMemset(tmpsens,0, sizeof(cufftDoubleComplex)*numK);
            
            
	  //            dosens<<<dimGrid_se,dimBlock_se>>>(tmpsens,tmp2,_ipk_we,_the_index,numP,numK);
	  //	  dosens<<<dimGrid_se[i],dimBlock_se>>>(tmpsens,tmp2,_segfil[i], _ipk_we,_the_index,numP,numK);
	  dosens<<<dimGrid_se[i],dimBlock_se>>>(tmpsens+segidx[i],
						tmp3, _segfil[i],
						_ipk_we[i], _the_index[i],
						numP, numK[i]);


	  cudaMemset(tmp1,0, sizeof(cufftDoubleComplex)*totsz_pad);            
            
            
	  //            backprojVX<<<dimGrid_vx,dimBlock_vx>>>(_bpidx,_bponset,_bpweight,_bpmidx,_bpsize,tmpsens, tmp1,numVox);
	  //	  backprojVX<<<dimGrid_vx,dimBlock_vx>>>(_bpidx,_bponset,_bpweight,_bpmidx,_bpsize,tmpsens, _segfil[i], tmp1,numVox);
	  backprojVX<<<dimGrid_vx[j], dimBlock_vx>>>(
						     _bpidx[i], 
						     _bponset[i], _bpweight[i], 
						     _bpmidx[i], _bpsize[i],
						     _meas + i*numKT+segidx[i], _segfil[i],
						     tmp1, numVox[i]);
              
                        
	  if (err=cufftExecZ2Z(plan, tmp1, tmp2, CUFFT_INVERSE) != CUFFT_SUCCESS)
            {
	      mexPrintf("3) cufft has failed with err %i \n",err);
	      return;
            }
	  downwind<<<dimGrid_dw,dimBlock_dw>>>(tmp3,tmp2, _sens + i*totsz, w, h, d, w_pad, h_pad, d_pad);                
         
        }
        
      cudaThreadSynchronize();
       
      GET_TIME(finish); AA_time += finish-start;
               
      GET_TIME(start);    
      dAAd = Dot_wrapper(_d, tmp3, dot_z_d, dot_z_h, totsz, dot_blocks, dot_threads) ;        
      //dAAd = cublasCdotc(totsz,(cuComplex*)_d,1,(cuComplex*)tmp3,1).x;           
     
      alpha = normrr/dAAd;        
      scpm<<<dimGrid_sq,dimBlock_sq>>>(_z,_d,alpha,totsz);
      scpm<<<dimGrid_sq,dimBlock_sq>>>(_r,tmp3,-alpha,totsz);                       	
      normrr2 = Dot_wrapper(_r, _r, dot_z_d, dot_z_h, totsz, dot_blocks, dot_threads) ;   
      //normrr2 = cublasCdotc(totsz,(cuComplex*)_r,1,(cuComplex*)_r,1).x;     
      beta = normrr2/normrr;
      normrr = normrr2;        
      scmultplus<<<dimGrid_sq,dimBlock_sq>>>(_d,_r,beta,totsz);     
        
      cudaThreadSynchronize();
        
      GET_TIME(finish);  cg_time += finish-start;
       
      if (sqrt(normrr/normrr0) < tol)
	break;       
        
            
      if (VERBOSE == 1)  
	mexPrintf("tol: %f\n",sqrt(normrr/normrr0));

      mexPrintf("."); mexEvalString("drawnow");
    
    }
         
   
  if (VERBOSE == 1)
    {
      mexPrintf("\n");        
      mexPrintf(" AA time: %f \n",AA_time);
      mexPrintf(" cg  time: %f \n",cg_time);
    }
        
  cudaMemcpy( res, _z, sizeof(cufftDoubleComplex)*totsz,cudaMemcpyDeviceToHost);
   

  //    cudaFreeHost(dot_z_h);     
  cudaFree(tmp1);
  cudaFree(tmp2);
  cudaFree(tmp3);
  cudaFree(_r); 
  cudaFree(_d);
  cudaFree(_z);
  cudaFree(_meas);
  cudaFree(_sens);
  cudaFree(_ipk_we);
  cudaFree(_the_index);
  cudaFree(tmpsens);
  cudaFree(_bpmidx);
  cudaFree(_bpweight);
  cudaFree(_bpsize);
  cudaFree(_bpidx);
  cudaFree(_bponset);    
    
  cufftDestroy(plan);
  free(bpsize);
  free(bponset);
 

  CUcontext  pctx ;
  cuCtxPopCurrent(&pctx);	
    
}













