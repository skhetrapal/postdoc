#!/usr/bin/env python3
import numpy as np
import time
import pickle
import sys
import getopt
from multiprocessing import Pool

    
class MPS:
    def __init__(self,svs,gamma1,gammas,gammaN):
        self.vidal = True
        if svs==[]:
            svs = [[] for i in range(len(gammas)+1)]
            self.vidal = False
        else:
            for s in svs:
                if s==[]:
                    self.vidal = False
                    
        self.svs = svs
        self.gamma1 = gamma1
        self.gammas = gammas
        self.gammaN = gammaN
        self.length = len(gammas)+2
        self.shape = [self.gamma1.shape]+[gamma.shape for gamma in self.gammas]+[self.gammaN.shape]
        
    def copy(self):
        return MPS(self.svs.copy(), self.gamma1.copy(),self.gammas.copy(),self.gammaN.copy())
        
    def contract(self):
        Sp = [self.gamma1.copy()]
        for n in range(len(self.gammas)):
            if self.svs[n]==[]:
                Sp.append(np.einsum('...i,ijk->...jk',Sp[-1],self.gammas[n]))
            else:
                Sp.append(np.einsum('...i,i,ijk->...jk',Sp[-1],self.svs[n],self.gammas[n]))
    
        if self.svs[-1]==[]:
            S = np.einsum('...i,ji->...j',Sp[-1],self.gammaN)
        else:
            S = np.einsum('...i,i,ji->...j',Sp[-1],self.svs[-1],self.gammaN)
        return S
            
    def dot(self,other):
        other = other.absorb_svs(1)
        this = self.absorb_svs(1)
        res = np.einsum('ij,ik->jk',this.gamma1,other.gamma1)
        for n in range(len(self.gammas)):
            res = np.einsum('ia,ijk->ajk',res,this.gammas[n])
            res = np.einsum('ajk,ajb->kb',res,other.gammas[n])
        
        return np.einsum('kb,jk,jb->',res,this.gammaN,other.gammaN)
        
    def conjugate(self):
        if self.vidal:
            return MPS([np.conjugate(sv) for sv in self.svs],np.conjugate(self.gamma1),[np.conjugate(gamma) for gamma in self.gammas],np.conjugate(self.gammaN))
        else:
            return MPS([],np.conjugate(self.gamma1),[np.conjugate(gamma) for gamma in self.gammas],np.conjugate(self.gammaN))
     
    def get_shape(self):
        return [self.gamma1.shape]+[gamma.shape for gamma in self.gammas]+[self.gammaN.shape]
     
    def absorb_svs(self,rls,ind=1):
        if rls==1 or ind<=0: #convert to right-canonical
            gamma1 = self.gamma1.copy()
            gammas = []
            for i in range(len(self.gammas)):
                gamma = self.gammas[i].copy()
                if self.svs[i]!=[]:
                    for k in range(self.svs[i].shape[0]):
                        gamma[k,:,:] = self.svs[i][k]*gamma[k,:,:]
                gammas.append(gamma)
                
            gammaN = self.gammaN.copy()
            if self.svs[-1]!=[]:
                for k in range(self.svs[-1].shape[0]):
                    gammaN[:,k] = self.svs[-1][k]*gammaN[:,k]
                
            return MPS([],gamma1,gammas,gammaN)
            
        if rls==-1 or ind>len(self.gammas): #convert to left-canonical
            gamma1 = self.gamma1.copy()
            if self.svs[0]!=[]:
                for k in range(self.svs[0].shape[0]):
                    gamma1[:,k] = self.svs[0][k]*gamma1[:,k]
                
            gammas = []
            for i in range(len(self.gammas)):
                gamma = self.gammas[i].copy()
                if self.svs[i+1]!=[]:
                    for k in range(self.svs[i+1].shape[0]):
                        gamma[:,:,k] = self.svs[i+1][k]*gamma[:,:,k]
                gammas.append(gamma)
                
            gammaN = self.gammaN.copy()
            return MPS([],gamma1,gammas,gammaN)
        
        if rls==0: #convert to mixed-canonical across ind
            gamma1 = self.gamma1.copy()
            if self.svs[0]!=[]:
                for k in range(self.svs[0].shape[0]):
                    gamma1[:,k] = self.svs[0][k]*gamma1[:,k]
                
            gammas = []
            for i in range(ind-1):
                gamma = self.gammas[i].copy()
                if self.svs[i+1]!=[]:
                    for k in range(self.svs[i+1].shape[0]):
                        gamma[:,:,k] = self.svs[i+1][k]*gamma[:,:,k]
                gammas.append(gamma)
            
            gammas.append(self.gammas[ind-1].copy())
            
            for i in range(ind,len(self.gammas)):
                gamma = self.gammas[i].copy()
                if self.svs[i]!=[]:
                    for k in range(self.svs[i].shape[0]):
                        gamma[k,:,:] = self.svs[i][k]*gamma[k,:,:]
                gammas.append(gamma)
                
            gammaN = self.gammaN.copy()
            if self.svs[-1]!=[]:
                for k in range(self.svs[-1].shape[0]):
                    gammaN[:,k] = self.svs[-1][k]*gammaN[:,k]
                   
            return MPS([],gamma1,gammas,gammaN)
            
    def split_svs(self):
        gamma1 = self.gamma1.copy()
        gammas = [gamma.copy() for gamma in self.gammas]
        if self.svs[0]!=[]:
            for k in range(self.svs[0].shape[0]):
                gamma1[:,k] = np.sqrt(self.svs[0][k])*gamma1[:,k]
                gammas[0][k,:,:] = np.sqrt(self.svs[0][k])*gammas[0][k,:,:]
        
        for i in range(len(self.gammas)-1):
            if self.svs[i]!=[]:
                for k in range(self.svs[i+1].shape[0]):
                    gammas[i][:,:,k] = np.sqrt(self.svs[i+1][k])*gammas[i][:,:,k]
                    gammas[i+1][k,:,:] = np.sqrt(self.svs[i+1][k])*gammas[i+1][k,:,:]
                    
        gammaN = self.gammaN.copy()
        if self.svs[-1]!=[]:
            for k in range(self.svs[-1].shape[0]):
                gammas[-1][:,:,k] = np.sqrt(self.svs[-1][k])*gammas[-1][:,:,k]
                gammaN[:,k] = np.sqrt(self.svs[-1][k])*gammaN[:,k]
                
        return MPS([],gamma1,gammas,gammaN)
            
    def to_string(self):
        MPSstring = str(self.gamma1)+'\n*\n'
        for n in range(len(self.gammas)):
            MPSstring += str(self.svs[n])+ '\n*\n' + str(self.gammas[n]) +'\n*\n'
        MPSstring += str(self.svs[-1])+ '\n*\n' + str(self.gammaN)
        return MPSstring
            
            
    def compress_site_SVD(self,r,d_max=-1,cutoff=(10**(-16))):
        if r==1:
            sz = self.gammas[0].shape
            S = 0
            if self.svs[0]==[]:
                S = np.einsum('ia,ajb->ijb',self.gamma1,self.gammas[0])
            else:
                S = np.einsum('ia,a,ajb->ijb',self.gamma1,self.svs[0],self.gammas[0])
            
            L,s,R = schmidt(S,1,d_max=d_max,cutoff=cutoff)
            gamma1 = L.copy()
            srank = s.shape[0]
            gammar = R.transpose().reshape((srank,sz[1],sz[2]))
            svs = self.svs.copy()
            svs[0]=s
            gammas = self.gammas.copy()
            gammas[0]=gammar
            gammaN = self.gammaN.copy()
            return MPS(svs,gamma1,gammas,gammaN)
        if r==(len(self.gammas)+1):
            sz1 = self.gammas[-1].shape
            S = 0
            if self.svs[-1]==[]:
                S = np.einsum('aib,jb->aij',self.gammas[-1],self.gammaN)
            else:
                S = np.einsum('aib,b,jb->aij',self.gammas[-1],self.svs[-1],self.gammaN)
            
            L,s,R = schmidt(S,2,d_max=d_max,cutoff=cutoff) # |S>=s1[a1]*|L1[:,a1]>|R1[:,a1]> (21)
            srank = s.shape[0]
            gammar = L.reshape((sz1[0],sz1[1],srank))
            svs = self.svs.copy()
            svs[-1]=s
            gammas = self.gammas.copy()
            gammas[-1]=gammar
            gammaN = R.copy()
            gamma1 = self.gamma1.copy()
            return MPS(svs,gamma1,gammas,gammaN)
        else:
            sz1 = self.gammas[r-2].shape
            sz2 = self.gammas[r-1].shape
            S = 0
            if self.svs[r-1]==[]:
                S = np.einsum('aib,bjc->aijc',self.gammas[r-2],self.gammas[r-1])
            else:
                S = np.einsum('aib,b,bjc->aijc',self.gammas[r-2],self.svs[r-1],self.gammas[r-1])
                
            L,s,R = schmidt(S,2,d_max=d_max,cutoff=cutoff)
            srank = s.shape[0]
            gammar1 = L.reshape((sz1[0],sz1[1],srank))
            gammar2 = R.transpose().reshape((srank,sz2[1],sz2[2]))
            svs = self.svs.copy()
            svs[r-1]=s
            gammas = self.gammas.copy()
            gammas[r-2]=gammar1
            gammas[r-1]=gammar2
            return MPS(svs,self.gamma1.copy(),gammas,self.gammaN.copy())
            
    def compress_SVD(self,d_max=-1,cutoff=(10**(-16))):
        mpsn = self.compress_site_SVD(1,d_max=d_max,cutoff=cutoff)
        for r in range(2,self.length):
            mpsn = mpsn.compress_site_SVD(r,d_max=d_max,cutoff=cutoff)
            
        return mpsn
            
    def __eq__(self,other):
        if other==None:
            return False
        if self.length!=other.length or self.vidal!=other.vidal:
            return False
        else:
            is_equal = True
            for k in range(len(self.svs)):
                is_equal = is_equal and self.svs[k].shape==other.svs[k].shape
                is_equal = is_equal and np.allclose(self.svs[k],other.svs[k])
                if not is_equal:
                    return False
                    
            for k in range(len(self.gammas)):
                is_equal = is_equal and self.svs[k].shape==other.svs[k].shape
                is_equal = is_equal and np.allclose(self.gammas[k],other.gammas[k])
                if not is_equal:
                    return False
                    
            return is_equal and np.allclose(self.gamma1,other.gamma1) and np.allclose(self.gammaN,other.gammaN)
            
    def __str__(self):
        return self.to_string()
        
    def apply_Ugate(self,U,r,d_max=-1,cutoff=(10**(-16))):
        print("apply_Ugate(r="+str(r),end="):")
        if r==0:
            # theta = np.einsum('ijlr,lb,b,brc,cd->ijd',U,self.gamma1,self.svs[0],self.gammas[0],np.diag(self.svs[1]))
            print("temp1",end="...",flush=True)
            temp1 = np.tensordot(self.gamma1*self.svs[0][None,:],self.gammas[0],axes=([1],[0])) #lrc
            print("temp2",end="...",flush=True)
            temp2 = temp1*self.svs[1][None,None,:] #lrd
            print("theta",end="...",flush=True)
            theta = np.tensordot(U,temp2,axes=([2,3],[0,1]))
            sz = theta.shape
            
            #M = theta.reshape(dims[0]*dims[1],dims[2]*dims[3])
            print("schmidt",end="...",flush=True)
            L,s,R = schmidt(theta,1,d_max=d_max,cutoff=cutoff)
            srank = s.shape[0]
            print(str(srank),end=".",flush=True)
            print("reshape",end="...",flush=True)
            gamma1 = L.reshape((sz[0],srank))
            gammar = R.transpose().reshape((srank,sz[1],sz[2]))
            # for k in range(sz[2]):
                # gammar[:,:,k] = gammar[:,:,k]/self.svs[1][k]
            print("/svs",end="...",flush=True)
            gammar = gammar/self.svs[1][None,None,:]
            print("done",end="",flush=True)
            self.svs[0] = s
            self.gamma1=gamma1
            self.gammas[0]=gammar
        elif r==len(self.gammas):
            # theta = np.einsum('ca,ijlr,alb,b,rb->cij',np.diag(self.svs[r-1]),U,self.gammas[r-1],self.svs[r],self.gammaN)
            print("temp1",end="...",flush=True)
            temp1 = self.svs[r-1][:,None,None]*self.gammas[r-1]
            print("temp2",end="...",flush=True)
            temp2 = np.tensordot(temp1*self.svs[r][None,None,:],self.gammaN,axes=([2],[1]))
            print("theta",end="...",flush=True)
            theta = np.tensordot(temp2,U,axes=([1,2],[2,3]))
            sz = theta.shape
            print("schmidt",end="...",flush=True)
            L,s,R = schmidt(theta,2,d_max=d_max,cutoff=cutoff)
            srank = s.shape[0]
            print(str(srank),end=".",flush=True)
            print("reshape",end="...",flush=True)
            gammal = L.reshape((sz[0],sz[1],srank))
            gammaN = R.reshape((sz[2],srank))
            # for k in range(sz[0]):
                # gammal[k,:,:] = gammal[k,:,:]/self.svs[r-1][k]
            print("/svs",end="...",flush=True)
            gammal = gammal/self.svs[r-1][:,None,None]
            print("done",end="",flush=True)
            self.svs[r] = s
            self.gammas[r-1]=gammal
            self.gammaN=gammaN
        else:
            # theta = np.einsum('ijlr,ua,alb,b,brc,cd->uijd',U,np.diag(self.svs[r-1]),self.gammas[r-1],self.svs[r],self.gammas[r],np.diag(self.svs[r+1]))
            print("temp1",end="...",flush=True)
            temp1 = self.svs[r-1][:,None,None]*self.gammas[r-1]
            print("temp2",end="...",flush=True)
            temp2 = np.tensordot(temp1*self.svs[r][None,None,:],self.gammas[r],axes=([2],[0]))
            print("theta",end="...",flush=True)
            theta = np.moveaxis(np.tensordot(temp2*self.svs[r+1][None,None,None,:],U,axes=([1,2],[2,3])),[1,2,3],[3,1,2])
            sz = theta.shape
            print("schmidt",end="...",flush=True)
            L,s,R = schmidt(theta,2,d_max=d_max,cutoff=cutoff)
            srank = s.shape[0]
            print(str(srank),end=".",flush=True)
            print("reshape",end="...",flush=True)
            gammal = L.reshape((sz[0],sz[1],srank))
            gammar = R.transpose().reshape((srank,sz[2],sz[3]))
            print("/svs",end="...",flush=True)
            gammal = gammal/self.svs[r-1][:,None,None]
            gammar = gammar/self.svs[r+1][None,None,:]
            print("done",end="",flush=True)
            self.svs[r] = s
            self.gammas[r-1]=gammal
            self.gammas[r]=gammar
            
    
    def get_EE(self,r):
        # S = 0
        # for k in range(len(self.svs[r-1])):
            # l = self.svs[r-1][k]**2
            # S += -l*np.log(l)
        l = self.svs[r-1]*self.svs[r-1]
        S = np.sum(-l*np.log(l))
        return S
        
        
    def get_reduced_density_matrix(self,l,r):
        C = []
        if l==1:
            C.append(np.einsum('ib,jc -> ijbc',self.gamma1.copy(),np.conjugate(self.gamma1.copy())))
        else:
            C.append(np.einsum('a,a,aib,ajc -> ...ijbc',self.svs[l-2],self.svs[l-2],self.gammas[l-2],np.conjugate(self.gammas[l-2])))
         
        for k in range(r-l-1):
            C.append(np.einsum('...ab,a,b,aic,bjd -> ...ijcd',C[-1],self.svs[l+k-1],self.svs[l+k-1],self.gammas[l+k-1],np.conjugate(self.gammas[l+k-1])))
               
        if r==(len(self.svs)+1):
            C.append(np.einsum('...ab,a,b,ia,jb -> ...ij',C[-1],self.svs[r-2],self.svs[r-2],self.gammaN,np.conjugate(self.gammaN)))
        else:
            C.append(np.einsum('...ab,a,b,aic,bjc,c,c -> ...ij',C[-1],self.svs[r-2],self.svs[r-2],self.gammas[r-2],np.conjugate(self.gammas[r-2]),self.svs[r-1],self.svs[r-1]))
            
        rho = C[-1]
        d = len(rho.shape)
        dests = [-k-1 for k in range(d//2)]
        dests.reverse()
        rho2 = np.moveaxis(rho,[k*2+1 for k in range(d//2)],dests)
        dims2 = rho2.shape
        d1 = 1
        d2 = 1
        for k in range(d//2):
            d1 = d1*dims2[k]
            d2 = d2*dims2[k+d//2]
        return rho2.reshape((d1,d2))
        
        
    def get_rhoAB(self,l,r):#(A=[1:l])U(B=[r:N])
        L = r-l-1
        print("M0",end="...",flush=True)
        M = np.swapaxes(np.tensordot(self.gammas[l-1]*self.svs[l][None,None,:],np.conjugate(self.gammas[l-1])*self.svs[l][None,None,:],axes=([1],[1])),1,2)#"aic,bid,c,d->abcd",gamma(l+1),gamma(l+1)*,lambda(l+1),lambda(l+1)
        for k in range(l+1,r-1):
            #tempEk = np.swapaxes(np.tensordot(self.gammas[k-1]*self.svs[k][None,None,:],np.conjugate(self.gammas[k-1])*self.svs[k][None,None,:],axes=([1],[1])),1,2)#"aic,bid,c,d->abcd",gamma(k+1),gamma(k+1)*,lambda(k+1),lambda(k+1)
            #M = np.tensordot(M,tempEk,axes=([2,3],[0,1])) #ijab,abcd->ijcd
            print("tmpM",end="...",flush=True)
            M = np.tensordot(M,self.gammas[k-1]*self.svs[k][None,None,:],axes=([2],[0]))#abcd,cir->abdir
            print("M",end="...",flush=True)
            M = np.tensordot(M,np.conjugate(self.gammas[k-1])*self.svs[k][None,None,:],axes=([2,3],[0,1])) #abdir,dil->abrl
            
        #print("M="+str(M.shape))
        print("L",end="...",flush=True)
        L = self.gamma1*self.svs[0][None,:] #ia,a->ia
        for k in range(1,l):
            print("L",end="...",flush=True)
            L = np.tensordot(L,self.gammas[k-1]*self.svs[k][None,None,:],axes=([k],[0]))#(...)a,aib,b->(...)ib, L,gamma(k+1),lambda(k+1)
         
        ds = L.shape
        #print("Lds="+str(ds))
        print("Lshape",end="...",flush=True)
        L = L.reshape((np.product(ds[0:-1]),ds[-1]))
        print("R",end="...",flush=True)
        R = np.swapaxes(self.gammaN,0,1) #ia->ai
        for k in range(len(self.gammas)+1,r-1,-1):
            print("R",end="...",flush=True)
            R = np.tensordot(self.gammas[k-2]*self.svs[k-1][None,None,:],R,axes=([2],[0]))#aib,b,b(...)->ai(...), gamma(k-1),lambda(k-1),R
            
        ds = R.shape
        #print("Rds="+str(ds))
        print("Rshape",end="...",flush=True)
        R = R.reshape((ds[0],np.product(ds[1:])))
        print("R,M->temp",end="...",flush=True)
        temp = np.tensordot(R,M,axes=([0],[2])) #cj,abcd ->jabd
        print("L,temp->temp",end="...",flush=True)
        temp = np.tensordot(L,temp,axes=([1],[1])) #ia,jabd->ijbd
        print("temp,L*->temp",end="...",flush=True)
        temp = np.tensordot(temp,np.conjugate(L),axes=([2],[1])) #ijbd,mb->ijdm
        print("temp,R*->temp",end="...",flush=True)
        temp = np.tensordot(temp,np.conjugate(R),axes=([2],[0]))  #ijdm,dn->ijmn
        shp = temp.shape
        #print("rho="+str(shp))
        print("reshape",end="...",flush=True)
        rhoAB = temp.reshape((shp[0]*shp[1],shp[2]*shp[3]))
        print("done",end=". ",flush=True)
        return rhoAB
        
        
    def get_IABrhoC(self,lA,rA,lB,rB):
        # SA = 0
        # if lA==1:
            # for k in range(len(self.svs[rA-1])):
                # l = self.svs[rA-1][k]**2
                # SA += -l*np.log(l)
        # else:      
        rhoA = self.get_reduced_density_matrix(lA,rA)
        dA,W = np.linalg.eigh(rhoA)
        SA = 0
        for k in range(dA.shape[0]):
            if dA[k]>0.0:
                SA = SA - dA[k]*np.log(dA[k])
    
        # SB = 0
        # if rB==(len(self.svs)+1):
            # for k in range(len(self.svs[lB-2])):
                # l = self.svs[lB-2][k]**2
                # SB += -l*np.log(l)
        # else:
        rhoB = self.get_reduced_density_matrix(lB,rB)
        dB,W = np.linalg.eigh(rhoB)
        SB = 0
        for k in range(dB.shape[0]):
            if dB[k]>0.0:
                SB = SB - dB[k]*np.log(dB[k])
         
        rhoAB = self.get_reduced_density_matrix(rA+1,lB-1)
        dAB,W = np.linalg.eigh(rhoAB)
        SAB = 0
        for k in range(dAB.shape[0]):
            if dAB[k]>0.0:
                SAB = SAB - dAB[k]*np.log(dAB[k])
               
        return SA,SB,SAB

    def get_IAB(self,l,r):
        print("get_I(A=[:"+str(l)+"],B=["+str(r)+":])",end=":",flush=True)
        rhoAB = self.get_rhoAB(l,r)
        print("eigs",end="...",flush=True)
        dAB,W = np.linalg.eigh(rhoAB)
        print("sumlog",end="...",flush=True)
        SAUB = 0#-np.sum(dAB*np.log(dAB))
        for k in range(dAB.shape[0]):
            if dAB[k]>0.0:
                SAUB = SAUB - dAB[k]*np.log(dAB[k])
                
        print("get_EE(A,B)",end="...",flush=True)
        SA = self.get_EE(l)
        SB = self.get_EE(r-1)
        print("done",end=".",flush=True)
        return SA,SB,SAUB
        
    def get_expectation_1site(self,O,r):
        if r==0:
            return np.einsum('ia,ij,jb,ab',np.conjugate(self.gamma1),O,self.gamma1,np.diag(self.svs[r]*self.svs[r]))
        elif r==len(self.svs):
            return np.einsum('ia,ij,jb,ab',np.conjugate(self.gammaN),O,self.gammaN,np.diag(self.svs[r-1]*self.svs[r-1]))
        else:
            return np.einsum('ra,ril,ij,ajb,lb',np.diag(self.svs[r-1]*self.svs[r-1]),np.conjugate(self.gammas[r-1]),O,self.gammas[r-1],np.diag(self.svs[r]*self.svs[r]))
        
        
        
        
class MPO:
    def __init__(self,svs,gamma1,gammas,gammaN):
        self.vidal = True
        if svs==[]:
            svs = [[] for i in range(len(gammas)+1)]
            self.vidal = False
        else:
            for s in svs:
                if s==[]:
                    self.vidal = False
                    
        self.svs = svs
        self.gamma1 = gamma1
        self.gammas = gammas
        self.gammaN = gammaN
        self.length = len(gammas)+2
        self.shape = [self.gamma1.shape]+[gamma.shape for gamma in self.gammas]+[self.gammaN.shape]
        
    def contract(self):
        Mp = [self.gamma1.copy()]
        for n in range(len(self.gammas)):
            if self.svs[n]==[]:
                Mp.append(np.einsum('...a,aijb->...ijb',Mp[-1],self.gammas[n]))
            else:
                Mp.append(np.einsum('...a,a,aijb->...ijb',Mp[-1],self.svs[n],self.gammas[n]))
    
        if self.svs[-1]==[]:
            M = np.einsum('...a,ija->...ij',Mp[-1],self.gammaN)
        else:
            M = np.einsum('...a,a,ija->...ij',Mp[-1],self.svs[-1],self.gammaN)
        return M
        
    def apply(self,mps):
        if self.vidal and mps.vidal:
            gamma1 = np.einsum('ijr,jb->irb',self.gamma1,mps.gamma1)
            sg1 = gamma1.shape
            gamma1 = gamma1.reshape((sg1[0],sg1[1]*sg1[2]))
            gammas = []
            for n in range(len(self.gammas)):
                gamma = np.einsum('lijr,ajb->lairb',self.gammas[n],mps.gammas[n])
                sg = gamma.shape
                sn = (sg[0]*sg[1],sg[2],sg[3]*sg[4])
                gammas.append(gamma.copy().reshape(sn))
            
            gammaN = np.einsum('ijl,ja->ila',self.gammaN,mps.gammaN)
            sgN = gammaN.shape
            gammaN = gammaN.reshape((sgN[0],sgN[1]*sgN[2]))
            
            svs = []
            for n in range(len(self.svs)):
                svn = np.array([[self.svs[n][i]*mps.svs[n][j] for j in range(len(mps.svs[n]))] for i in range(len(self.svs[n]))])
                ls = len(mps.svs[n])*len(self.svs[n])
                svs.append(svn.copy().reshape((ls,)))
                
            return MPS(svs,gamma1,gammas,gammaN) 
            
        else:
            mpo = self.absorb_svs(1)
            mps = mps.absorb_svs(1)
            gamma1 = np.einsum('ijr,jb->irb',mpo.gamma1,mps.gamma1)
            sg1 = gamma1.shape
            gamma1 = gamma1.reshape((sg1[0],sg1[1]*sg1[2]))
            gammas = []
            for n in range(len(self.gammas)):
                gamma = np.einsum('lijr,ajb->lairb',mpo.gammas[n],mps.gammas[n])
                sg = gamma.shape
                sn = (sg[0]*sg[1],sg[2],sg[3]*sg[4])
                gammas.append(gamma.copy().reshape(sn))
            
            gammaN = np.einsum('ijl,ja->ila',mpo.gammaN,mps.gammaN)
            sgN = gammaN.shape
            gammaN = gammaN.reshape((sgN[0],sgN[1]*sgN[2]))
            return MPS([],gamma1,gammas,gammaN)
        
    def compose(self,other):
        if self.vidal and other.vidal:
            gamma1 = np.einsum('ijr,jkb->ikrb',self.gamma1,other.gamma1)
            sg1 = gamma1.shape
            gamma1 = gamma1.reshape((sg1[0],sg1[1],sg1[2]*sg1[3]))
            gammas = []
            for n in range(len(self.gammas)):
                gamma = np.einsum('lijr,ajkb->laikrb',self.gammas[n],other.gammas[n])
                sg = gamma.shape
                sn = (sg[0]*sg[1],sg[2],sg[3],sg[4]*sg[5])
                gammas.append(gamma.copy().reshape(sn))
            
            gammaN = np.einsum('ijl,jkb->iklb',self.gammaN,other.gammaN)
            sgN = gammaN.shape
            gammaN = gammaN.reshape((sgN[0],sgN[1],sgN[2]*sgN[3]))
            
            svs = []
            for n in range(len(self.svs)):
                svn = np.array([[self.svs[n][i]*other.svs[n][j] for j in range(len(other.svs[n]))] for i in range(len(self.svs[n]))])
                svs.append(svn.copy().reshape((svn.shape[0]*svn.shape[1],)))
                
            return MPO(svs,gamma1,gammas,gammaN)
            
        else:
            this = self.absorb_svs(1)
            other = other.absorb_svs(1)
            gamma1 = np.einsum('ijr,jkb->ikrb',this.gamma1,other.gamma1)
            sg1 = gamma1.shape
            gamma1 = gamma1.reshape((sg1[0],sg1[1],sg1[2]*sg1[3]))
            gammas = []
            for n in range(len(self.gammas)):
                gamma = np.einsum('lijr,ajkb->laikrb',this.gammas[n],other.gammas[n])
                sg = gamma.shape
                sn = (sg[0]*sg[1],sg[2],sg[3],sg[4]*sg[5])
                gammas.append(gamma.copy().reshape(sn))
            
            gammaN = np.einsum('ijl,jkb->iklb',this.gammaN,other.gammaN)
            sgN = gammaN.shape
            gammaN = gammaN.reshape((sgN[0],sgN[1],sgN[2]*sgN[3]))
            return MPO([],gamma1,gammas,gammaN)
            
        
    def absorb_svs(self,rls,ind=1):
        if rls==1 or ind<=0: #convert to right-canonical
            gamma1 = self.gamma1.copy()
            gammas = []
            for i in range(len(self.gammas)):
                gamma = self.gammas[i].copy()
                if self.svs[i]!=[]:
                    for k in range(self.svs[i].shape[0]):
                        gamma[k,:,:,:] = self.svs[i][k]*gamma[k,:,:,:]
                        
                gammas.append(gamma)
                
            gammaN = self.gammaN.copy()
            if self.svs[-1]!=[]:
                for k in range(self.svs[-1].shape[0]):
                    gammaN[:,:,k] = self.svs[-1][k]*gammaN[:,:,k]
                
            return MPO([],gamma1,gammas,gammaN)
            
        if rls==-1 or ind>len(self.gammas): #convert to left-canonical
            gamma1 = self.gamma1.copy()
            if self.svs[0]!=[]:
                for k in range(self.svs[0].shape[0]):
                    gamma1[:,:,k] = self.svs[0][k]*gamma1[:,:,k]
                
            gammas = []
            for i in range(len(self.gammas)):
                gamma = self.gammas[i].copy()
                if self.svs[i+1]!=[]:
                    for k in range(self.svs[i+1].shape[0]):
                        gamma[:,:,:,k] = self.svs[i+1][k]*gamma[:,:,:,k]
                gammas.append(gamma)
                
            gammaN = self.gammaN.copy()
            return MPO([],gamma1,gammas,gammaN)
        
        if rls==0: #convert to mixed-canonical across ind
            gamma1 = self.gamma1.copy()
            if self.svs[0]!=[]:
                for k in range(self.svs[0].shape[0]):
                    gamma1[:,:,k] = self.svs[0][k]*gamma1[:,:,k]
                
            gammas = []
            for i in range(ind-1):
                gamma = self.gammas[i].copy()
                if self.svs[i+1]!=[]:
                    for k in range(self.svs[i+1].shape[0]):
                        gamma[:,:,:,k] = self.svs[i+1][k]*gamma[:,:,:,k]
                gammas.append(gamma)
            
            gammas.append(self.gammas[ind-1].copy())
            
            for i in range(ind,len(self.gammas)):
                gamma = self.gammas[i].copy()
                if self.svs[i]!=[]:
                    for k in range(self.svs[i].shape[0]):
                        gamma[k,:,:,:] = self.svs[i][k]*gamma[k,:,:,:]
                gammas.append(gamma)
                
            gammaN = self.gammaN.copy()
            if self.svs[-1]!=[]:
                for k in range(self.svs[-1].shape[0]):
                    gammaN[:,:,k] = self.svs[-1][k]*gammaN[:,:,k]
                
            return MPO([],gamma1,gammas,gammaN)
            
    def split_svs(self):
        gamma1 = self.gamma1.copy()
        gammas = [gamma.copy() for gamma in self.gammas]
        if self.svs[0]!=[]:
            for k in range(self.svs[0].shape[0]):
                gamma1[:,:,k] = np.sqrt(self.svs[0][k])*gamma1[:,:,k]
                gammas[0][k,:,:,:] = np.sqrt(self.svs[0][k])*gammas[0][k,:,:,:]
        
        for i in range(len(self.gammas)-1):
            if self.svs[i]!=[]:
                for k in range(self.svs[i+1].shape[0]):
                    gammas[i][:,:,:,k] = np.sqrt(self.svs[i+1][k])*gammas[i][:,:,:,k]
                    gammas[i+1][k,:,:,:] = np.sqrt(self.svs[i+1][k])*gammas[i+1][k,:,:,:]
                    
        gammaN = self.gammaN.copy()
        if self.svs[-1]!=[]:
            for k in range(self.svs[-1].shape[0]):
                gammas[-1][:,:,:,k] = np.sqrt(self.svs[-1][k])*gammas[-1][:,:,:,k]
                gammaN[:,:,k] = np.sqrt(self.svs[-1][k])*gammaN[:,:,k]
                
        return MPO([],gamma1,gammas,gammaN)
     
    def to_string(self):
        MPSstring = str(self.gamma1)+'\n*\n'
        for n in range(len(self.gammas)):
            MPSstring += str(self.svs[n])+ '\n*\n' + str(self.gammas[n]) +'\n*\n'
        MPSstring += str(self.svs[-1])+ '\n*\n' + str(self.gammaN)
        return MPSstring
        
    def compress_site_SVD(self,r,d_max=-1,cutoff=(10**(-16))):
        if r==1:
            sz1 = self.gamma1.shape
            sz2 = self.gammas[0].shape
            S = 0
            if self.svs[0]==[]:
                S = np.einsum('ija,aklb->ijklb',self.gamma1,self.gammas[0])
            else:
                S = np.einsum('ija,a,aklb->ijklb',self.gamma1,self.svs[0],self.gammas[0])
            
            L,s,R = schmidt(S,2,d_max=d_max,cutoff=cutoff)
            srank = s.shape[0]
            gamma1 = L.reshape((sz1[0],sz1[1],srank))
            gammar = R.transpose().reshape((srank,sz2[1],sz2[2],sz2[3]))
            svs = self.svs.copy()
            svs[0]=s
            gammas = self.gammas.copy()
            gammas[0]=gammar
            gammaN = self.gammaN.copy()
            return MPO(svs,gamma1,gammas,gammaN)
        if r==(len(self.gammas)+1):
            sz1 = self.gammas[-1].shape
            sz2 = self.gammaN.shape
            S = 0
            if self.svs[-1]==[]:
                S = np.einsum('aijb,klb->aijkl',self.gammas[-1],self.gammaN)
            else:
                S = np.einsum('aijb,b,klb->aijkl',self.gammas[-1],self.svs[-1],self.gammaN)
            
            L,s,R = schmidt(S,3,d_max=d_max,cutoff=cutoff)
            srank = s.shape[0]
            gammar = L.reshape((sz1[0],sz1[1],sz1[2],srank))
            svs = self.svs.copy()
            svs[-1]=s
            gammas = self.gammas.copy()
            gammas[-1]=gammar
            gammaN = R.reshape((sz2[0],sz2[1],srank))
            return MPO(svs,self.gamma1.copy(),gammas,gammaN)
        else:
            sz1 = self.gammas[r-2].shape
            sz2 = self.gammas[r-1].shape
            S = 0
            if self.svs[r-1]==[]:
                S = np.einsum('aijb,bklc->aijklc',self.gammas[r-2],self.gammas[r-1])
            else:
                S = np.einsum('aijb,b,bklc->aijklc',self.gammas[r-2],self.svs[r-1],self.gammas[r-1])
                
            L,s,R = schmidt(S,3,d_max=d_max,cutoff=cutoff)
            srank = s.shape[0]
            gammar1 = L.reshape((sz1[0],sz1[1],sz1[2],srank))
            gammar2 = R.transpose().reshape((srank,sz2[1],sz2[2],sz2[3]))
            svs = self.svs.copy()
            svs[r-1]=s
            gammas = self.gammas.copy()
            gammas[r-2]=gammar1
            gammas[r-1]=gammar2
            return MPO(svs,self.gamma1.copy(),gammas,self.gammaN.copy())
            
    def compress_SVD(self,d_max=-1,cutoff=(10**(-16))):
        mpon = self.compress_site_SVD(1,d_max=d_max,cutoff=cutoff)
        for r in range(2,self.length):
            mpon = mpon.compress_site_SVD(r,d_max=d_max,cutoff=cutoff)
            
        return mpon
  
    def __eq__(self,other):
        if self.length!=other.length or self.vidal!=other.vidal:
            return False
        else:
            is_equal = True
            for k in range(len(self.svs)):
                is_equal = is_equal and self.svs[k].shape==other.svs[k].shape
                is_equal = is_equal and np.allclose(self.svs[k],other.svs[k])
                if not is_equal:
                    return False
                    
            for k in range(len(self.gammas)):
                is_equal = is_equal and self.svs[k].shape==other.svs[k].shape
                is_equal = is_equal and np.allclose(self.gammas[k],other.gammas[k])
                if not is_equal:
                    return False
                    
            return is_equal and np.allclose(self.gamma1,other.gamma1) and np.allclose(self.gammaN,other.gammaN)
            
    def __str__(self):
        return self.to_string()
            
        

def schmidt_full(S,r): #schmidt-decomposition
    sz = S.shape
    N = len(sz)
    m = 1
    n = 1
    for k in range(r):
        m*=sz[k]
    for k in range(r,N):
        n*=sz[k]
        
    M = S.copy().reshape((m,n))
    u,s,vh = np.linalg.svd(M,full_matrices=True)
    #|u[:,k]>=left schmidt-vectors
    #s[k]=schmidt-coefficients
    #|vh[k,:]>=right schmidt-vectors
    return u,s,np.transpose(vh)
    
   
def cMPS_full(S): #exact canonical MPS representation of arbitrary state (according to 1306.2164)
    sz = S.shape
    N = len(sz)
    dtot = 1
    for k in range(N):
        dtot*=sz[k]
        
    L1,s1,R1 = schmidt_full(S,1) # |S>=s1[a1]*|L1[:,a1]>|R1[:,a1]> (21)
    gamma1 = L1.copy() # left schmidt-vectors |L1[:,a1]> = gamma1[i1,a1]*|i1>
    ns = s1.shape[0]
    
    svs = [s1]
    gammas = []
    dR = dtot//sz[0]
    for n in range(1,N-1):
        dr = dR//sz[n]
        W = [[R1[dr*s:(dr*(s+1)),k] for s in range(sz[n])] for k in range(ns)] # decompose |R1[:,a1]>=|i2>|W[a1][i2]> (23)
    
        L2,s2,R2 = schmidt_full(S,n+1)
        
        sfs = np.ones((dr,),dtype='complex')
        sfs[0:s2.shape[0]]= s2 +(s2==(0+0j)) #avoid zero-division (zero-schmidt-coefficients don't contribute)
        
        gamma = [[(np.matmul(np.conjugate(np.transpose(R2)),w)/sfs)[0:s2.shape[0]] for w in ws] for ws in W] # decompose |W[a1][i2]> = gamma[a1][i2][a2]*s2[a2]*|R2[:,a2]> (24)
        # (write |W2[a1][i2]> in basis |R2[:,a2]> --> change of basis-matrix = np.conjugate(np.transpose(R2)))
        R1 = R2
        ns = s2.shape[0]
        svs.append(s2)
        gammas.append(np.array(gamma))
        dR = dr
    
    gammaN = R1
    return svs,gamma1,gammas,gammaN



def schmidt(S,r,d_max=-1,cutoff=(10**(-16))): #schmidt-decomposition
    sz = S.shape
    N = len(sz)
    m = 1
    n = 1
    for k in range(r):
        m*=sz[k]
    for k in range(r,N):
        n*=sz[k]
    
    k = min(m,n)    
    M = S.copy().reshape((m,n))
    u,s,vh = np.linalg.svd(M,full_matrices=False)
    #|u[:,k]>=left schmidt-vectors
    #s[k]=schmidt-coefficients
    #|vh[k,:]>=right schmidt-vectors
    
    #discard neglegible (<cutoff) terms
    if d_max==-1 or d_max>k:
        d_max = k
        
    k_cut = d_max
    for i in range(d_max-1,-1,-1):
        k_cut = i+1
        if s[i]>cutoff:
            break
            
    # if (k_cut<k) and (s[k_cut-1]>(10**(-10))):
        # print("Schmidt k_cut="+str(k_cut)+"<"+str(k)+" "+str(s[k_cut]))
    u = u[:,0:k_cut]
    vh = vh[0:k_cut,:]
    s = s[0:k_cut]
    
    s = s / np.sqrt(np.sum(s*s))
    # if d_max==-1: #keep all
        # return u,s,np.transpose(vh)
    
    # if d_max<k_cut:
        # truncate decomposition
        # u = u[:,0:d_max]
        # vh = vh[0:d_max,:]
        # s = s[0:d_max]
        
    return u,s,np.transpose(vh)
 
def cMPS(S,d_max=-1,cutoff=(10**(-16))): #bounded bond-dimension canonical MPS representation of arbitrary state (according to 1306.2164)
    sz = S.shape
    N = len(sz)
    dtot = 1
    for k in range(N):
        dtot*=sz[k]
        
    L1,s1,R1 = schmidt(S,1,d_max=d_max,cutoff=cutoff) # |S>=s1[a1]*|L1[:,a1]>|R1[:,a1]> (21)
    gamma1 = L1.copy() # left schmidt-vectors |L1[:,a1]> = gamma1[i1,a1]*|i1>
    s1rank = s1.shape[0]
    
    svs = [s1]
    gammas = []
    dR = dtot//sz[0]
    for n in range(1,N-1):
        dr = dR//sz[n]
        W = [[R1[dr*s:(dr*(s+1)),k] for s in range(sz[n])] for k in range(s1rank)] # decompose |R1[:,a1]>=|i2>|W[a1][i2]> (23)
    
        L2,s2,R2 = schmidt(S,n+1,d_max=d_max,cutoff=cutoff)
        s2rank = s2.shape[0]
        
        sfs = np.ones((min(dr,s2rank),),dtype='complex')
        sfs[0:s2rank]= s2 +(s2==(0+0j)) #avoid zero-division (zero-schmidt-coefficients don't contribute)
        
        gamma = [[(np.matmul(np.conjugate(np.transpose(R2)),w)/sfs)[0:s2rank] for w in ws] for ws in W] # decompose |W[a1][i2]> = gamma[a1][i2][a2]*s2[a2]*|R2[:,a2]> (24)
        # (write |W2[a1][i2]> in basis |R2[:,a2]> --> change of basis-matrix = np.conjugate(np.transpose(R2)))
        R1 = R2
        s1rank = s2rank
        svs.append(s2)
        gammas.append(np.array(gamma))
        dR = dr
    
    gammaN = R1
    return MPS(svs,gamma1,gammas,gammaN)
    

def cMPO(M,d_max=-1,cutoff=(10**(-16))):
    sz = M.shape #(i1,o1,i2,o2,i3,o3,...iN,oN)
    szc = tuple([sz[2*k]*sz[2*k+1] for k in range(len(sz)//2)])
    cmps = cMPS(M.copy().reshape(szc),d_max=d_max,cutoff=cutoff)
    svs,gamma1,gammas,gammaN = (cmps.svs,cmps.gamma1,cmps.gammas,cmps.gammaN)
    sg1 = gamma1.shape
    gamma1 = gamma1.reshape((sz[0],sz[1],sg1[1]))
    gammasn = []
    for k in range(len(gammas)):
        sg = gammas[k].shape
        gammas[k] = gammas[k].copy().reshape((sg[0],sz[2*(k+1)],sz[2*(k+1)+1],sg[2]))
    sgN = gammaN.shape
    gammaN = gammaN.reshape((sz[-2],sz[-1],sgN[1]))
    return MPO(svs,gamma1,gammas,gammaN)

    
def contract_cMPS(svs,gamma1,gammas,gammaN): #contraction of MPS
    Sp = [gamma1]
    for n in range(len(gammas)):
        Sp.append(np.einsum('...i,i,ijk->...jk',Sp[-1],svs[n],gammas[n]))
        
    S = np.einsum('...i,i,ji->...j',Sp[-1],svs[-1],gammaN)
    return S

   
def cMPS2string(svs,gamma1,gammas,gammaN):
    MPSstring = str(gamma1)+'\n*\n'
    for n in range(len(gammas)):
        MPSstring += str(svs[n])+ '\n*\n' + str(gammas[n]) +'\n*\n'
    MPSstring += str(svs[-1])+ '\n*\n' + str(gammaN)
    return MPSstring

    
def randomsparse(dims,density=0.1):
    dtot = 1
    for d in dims:
        dtot=dtot*d
    
    r = np.random.random_sample((dtot,))*2-np.ones((dtot,))+(np.random.random_sample((dtot,))*2-np.ones((dtot,)))*1j
    is_all0 = True
    for k in range(dtot):
        v = np.random.random()
        if v>=density:
            r[k]=0
        else:
            is_all0 = False
    
    if is_all0:
        r[np.random.randint(dtot)]=np.random.random()*2-1 + (np.random.random()*2-1)*1j
        
    return r.reshape(dims)
    
def randomproductstate(dims):
    dtot = 1
    for d in dims:
        dtot=dtot*d
    r = np.zeros((dtot,),dtype=complex)
    c = np.random.random()*2-1 + (np.random.random()*2-1)*1j
    r[np.random.randint(dtot)] = c/np.abs(c)
    return r.reshape(dims)
    
def Ising_fullH_MPS(mps0,delta_t,Nsteps,h,g,J,func = lambda x : x.get_EE(3),d_max=-1,cutoff=(10**(-16))):
    h_bar = 1
    coupling = h_bar*h_bar*J#/4.0
    gfield = h_bar*g#/2.0
    hfield = h_bar*h#/2.0
    print("(h,g,J)="+str((h,g,J)))
    L = len(mps0.svs)+1
    
    H_field = np.array([[hfield,gfield],[gfield,-hfield]])
    H_coupling = np.diag([J,-J,-J,J])
    H_start = np.diag([-J,J])
    H_end = np.diag([-J,J])
    H_full = np.kron(H_start+H_field,np.diag([1 for k in range(2**(L-1))])) + np.kron(np.diag([1 for k in range(2**(L-1))]),H_end+H_field)
    H_full = H_full + np.kron(H_coupling,np.diag([1 for k in range(2**(L-2))])) + np.kron(np.diag([1 for k in range(2**(L-2))]),H_coupling)
    for r in range(1,L-1):
        H_full = H_full + np.kron(np.diag([1 for k in range(2**r)]),np.kron(H_coupling,np.diag([1 for k in range(2**(L-r-2))])))
    for r in range(1,L-1):
        H_full = H_full + np.kron(np.diag([1 for k in range(2**r)]),np.kron(H_field,np.diag([1 for k in range(2**(L-r-1))])))
        
    d,W = np.linalg.eigh(H_full)
    
    data = [func(mps0)]
    mps = mps0
    S = mps.contract().reshape((2**L,))
    
    for k in range(Nsteps):
        print("Step "+str(k),end="\r")
        D = np.diag(np.exp(-1j/h_bar*delta_t*(k+1)*d))
        U = np.matmul(W,np.matmul(D,np.transpose(np.conjugate(W))))
        S = np.matmul(U,S)
        mps = cMPS(S.reshape(tuple([2]*L)))
        data.append(func(mps))
    
    return mps,data
    
def Ising_fullH_EE(S0,delta_t,Nsteps,h,g,J,d_max=-1,cutoff=(10**(-16))):
    h_bar = 1
    coupling = h_bar*h_bar*J#/4.0
    gfield = h_bar*g#/2.0
    hfield = h_bar*h#/2.0
    print("(h,g,J)="+str((h,g,J)))
    L = S0.shape[0].bit_length()-1
    print("L="+str(L))
    
    H_field = np.array([[hfield,gfield],[gfield,-hfield]])
    H_coupling = np.diag([J,-J,-J,J])
    H_start = np.diag([-J,J])
    H_end = np.diag([-J,J])
    H_full = np.kron(H_start+H_field,np.diag([1 for k in range(2**(L-1))])) + np.kron(np.diag([1 for k in range(2**(L-1))]),H_end+H_field)
    H_full = H_full + np.kron(H_coupling,np.diag([1 for k in range(2**(L-2))])) + np.kron(np.diag([1 for k in range(2**(L-2))]),H_coupling)
    for r in range(1,L-2):
        H_full = H_full + np.kron(np.diag([1 for k in range(2**r)]),np.kron(H_coupling,np.diag([1 for k in range(2**(L-r-2))])))
    for r in range(1,L-1):
        H_full = H_full + np.kron(np.diag([1 for k in range(2**r)]),np.kron(H_field,np.diag([1 for k in range(2**(L-r-1))])))
        
    d,W = np.linalg.eigh(H_full)
    
    S = S0.copy()
    d2 = 2**(L//2)
    L,s,R = schmidt(S.reshape((d2,d2)),1,d_max=d_max,cutoff=cutoff)
    l = s**2
    data = [np.sum(-l*np.log(l))]
    
    for k in range(Nsteps):
        print("Step "+str(k),end="\r")
        D = np.diag(np.exp(-1j/h_bar*delta_t*(k+1)*d))
        U = np.matmul(W,np.matmul(D,np.transpose(np.conjugate(W))))
        S = np.matmul(U,S0)
        L,s,R = schmidt(S.reshape((d2,d2)),1,d_max=d_max,cutoff=cutoff)
        l = s**2
        data.append(np.sum(-l*np.log(l)))
    
    return S,data                                                                        
    
#def apply_Ugate1(U,gamma1,lambdaC,gammaR,lambdaR,d_max,cutoff): #self,U,r,d_max=-1,cutoff=(10**(-16))):           
#def apply_UgateN(U,lambdaL,gammaL,lambdaC,gammaN,d_max,cutoff): #self,U,r,d_max=-1,cutoff=(10**(-16))):
    
        
def apply_Ugate(inpt): #self,U,r,d_max=-1,cutoff=(10**(-16))):
    U,lambdaL,gammaL,lambdaC,gammaR,lambdaR,d_max,cutoff = inpt
    if lambdaL==[]:
        # theta = np.einsum('ijlr,lb,b,brc,cd->ijd',U,self.gamma1,self.svs[0],self.gammas[0],np.diag(self.svs[1]))
        temp1 = np.tensordot(gammaL*lambdaC[None,:],gammaR,axes=([1],[0])) #lrc
        temp2 = temp1*lambdaR[None,None,:] #lrd
        theta = np.tensordot(U,temp2,axes=([2,3],[0,1]))
        sz = theta.shape
        
        #M = theta.reshape(dims[0]*dims[1],dims[2]*dims[3])
        L,s,R = schmidt(theta,1,d_max=d_max,cutoff=cutoff)
        srank = s.shape[0]
        gammal = L.reshape((sz[0],srank))
        gammar = R.transpose().reshape((srank,sz[1],sz[2]))
        # for k in range(sz[2]):
            # gammar[:,:,k] = gammar[:,:,k]/self.svs[1][k]
        gammar = gammar/lambdaR[None,None,:]
            
        # self.svs[0] = s
        # self.gamma1=gamma1
        # self.gammas[0]=gammar
        return gammal,s,gammar
    elif lambdaR==[]:
        # theta = np.einsum('ca,ijlr,alb,b,rb->cij',np.diag(self.svs[r-1]),U,self.gammas[r-1],self.svs[r],self.gammaN)
        temp1 = lambdaL[:,None,None]*gammaL
        temp2 = np.tensordot(temp1*lambdaC[None,None,:],gammaR,axes=([2],[1]))
        theta = np.tensordot(temp2,U,axes=([1,2],[2,3]))
        sz = theta.shape
        L,s,R = schmidt(theta,2,d_max=d_max,cutoff=cutoff)
        srank = s.shape[0]
        gammal = L.reshape((sz[0],sz[1],srank))
        gammar = R.reshape((sz[2],srank))
        # for k in range(sz[0]):
            # gammal[k,:,:] = gammal[k,:,:]/self.svs[r-1][k]
        gammal = gammal/lambdaL[:,None,None]
            
        # self.svs[r] = s
        # self.gammas[r-1]=gammal
        # self.gammaN=gammaN
        return gammal,s,gammar
    else:
        # theta = np.einsum('ijlr,ua,alb,b,brc,cd->uijd',U,np.diag(self.svs[r-1]),self.gammas[r-1],self.svs[r],self.gammas[r],np.diag(self.svs[r+1]))
        temp1 = lambdaL[:,None,None]*gammaL
        temp2 = np.tensordot(temp1*lambdaC[None,None,:],gammaR,axes=([2],[0]))
        theta = np.moveaxis(np.tensordot(temp2*lambdaR[None,None,None,:],U,axes=([1,2],[2,3])),[1,2,3],[3,1,2])
        sz = theta.shape
        L,s,R = schmidt(theta,2,d_max=d_max,cutoff=cutoff)
        srank = s.shape[0]
        gammal = L.reshape((sz[0],sz[1],srank))
        gammar = R.transpose().reshape((srank,sz[2],sz[3]))
        
        # for k in range(sz[0]):
            # gammal[k,:,:] = gammal[k,:,:]/self.svs[r-1][k]
        # for k in range(sz[3]):
            # gammar[:,:,k] = gammar[:,:,k]/self.svs[r+1][k]
        gammal = gammal/lambdaL[:,None,None]
        gammar = gammar/lambdaR[None,None,:]
            
        # self.svs[r] = s
        # self.gammas[r-1]=gammal
        # self.gammas[r]=gammar
        return gammal,s,gammar
        
def apply_Ugates(inputs):
    results = []
    for input in inputs:
        results.append(apply_Ugate(input))
    return results
       
def splitlistN(lst,N):
    results = [[] for k in range(N)]
    for k in range(len(lst)):
        n = k%N
        results[n].append(lst[k])
        
    return results
    
def mergelistN(lsts,N):
    results = []
    L = 0
    for k in range(len(lsts)):
        L = L+len(lsts[k])
    for k in range(L):
        n = k%N
        results.append(lsts[n][k//N])
        
    return results
     
def Ising_TEBD_asymm_edge(mps0,delta_t,Nsteps,h,g,J,func=None,d_max=-1,cutoff=(10**(-16)),NUMTHREADS=4,LOGFILE=None):#assym + add edge J-terms
    h_bar = 1
    coupling = h_bar*h_bar*J#/4.0
    gfield = h_bar*g#/2.0
    hfield = h_bar*h#/2.0
    
    print("(h,g,J)="+str((h,g,J)))
    
    #H_odd = np.array([[coupling,0,0,0],[0,-coupling,0,0],[0,0,-coupling,0],[0,0,0,coupling]])
    expHodd = np.exp(-1j/h_bar*delta_t*coupling)
    expHoddinv = 1/expHodd
    U_odd = np.diag([expHodd,expHoddinv,expHoddinv,expHodd]).reshape((2,2,2,2))
    
    H_even = np.array([[2*hfield+coupling,gfield,gfield,0],[gfield,-coupling,0,gfield],[gfield,0,-coupling,gfield],[0,gfield,gfield,coupling-2*hfield]])
    d,W = np.linalg.eigh(H_even)
    D = np.diag(np.exp(-1j/h_bar*delta_t*d))
    U_even = np.matmul(W,np.matmul(D,np.transpose(np.conjugate(W)))).reshape((2,2,2,2))
    
    H_start = np.array([[2*hfield,gfield,gfield,0],[gfield,-2*coupling,0,gfield],[gfield,0,0,gfield],[0,gfield,gfield,2*coupling-2*hfield]])
    d,W = np.linalg.eigh(H_start)
    D = np.diag(np.exp(-1j/h_bar*delta_t*d))
    U_start= np.matmul(W,np.matmul(D,np.transpose(np.conjugate(W)))).reshape((2,2,2,2))
    
    H_end = np.array([[2*hfield,gfield,gfield,0],[gfield,0,0,gfield],[gfield,0,-2*coupling,gfield],[0,gfield,gfield,2*coupling-2*hfield]])
    d,W = np.linalg.eigh(H_end)
    D = np.diag(np.exp(-1j/h_bar*delta_t*d))
    U_end = np.matmul(W,np.matmul(D,np.transpose(np.conjugate(W)))).reshape((2,2,2,2))
    
    mps = mps0.copy()
    n_sites = len(mps.gammas)+2
    n2 = n_sites//2
    no = n2
    if n_sites%2==0:
       no -= 1
      
    if func==None:
        func = lambda x : x.get_EE(n2)                                                            
    print(str((n2,no)))
        
    data = [func(mps)]
    print(str(mps.get_shape()))
    print("data="+str(data[-1]))
    
    if NUMTHREADS<1:
        for k in range(Nsteps):
            print("step:"+str(k),end="\r",flush=True)
            if LOGFILE!=None:
                log_file = open(LOGFILE,"a+")
                log_file.write("step:"+str(k)+":"+str(mps.get_shape())+"\n")
                log_file.close() 
            for n in range(1,n2-1):
                print("step:"+str(k),end=":",flush=True)
                mps.apply_Ugate(U_even,2*n,d_max=d_max,cutoff=cutoff)
                print("done",end="\r                                                                                             \r",flush=True)
            print("step:"+str(k),end=":",flush=True)    
            mps.apply_Ugate(U_end,2*(n2-1),d_max=d_max,cutoff=cutoff)
            print("done",end="\r                                                                                           \r",flush=True)
            print("step:"+str(k),end=":",flush=True)
            mps.apply_Ugate(U_start,0,d_max=d_max,cutoff=cutoff)
            print("done",end="\r                                                                                           \r",flush=True)
            print("step:"+str(k),end=":",flush=True)
            for n in range(0,no):
                mps.apply_Ugate(U_odd,2*n+1,d_max=d_max,cutoff=cutoff)
                print("done",end="\r                                                                                             \r",flush=True)
                print("step:"+str(k),end=":",flush=True)
            print("computing IAB",end=":",flush=True)
            data.append(func(mps))
            print("done",end="\r                                                                                             \r",flush=True)
            print(str(mps.get_shape()))
        return mps,data
    
    pool_N = Pool(processes=NUMTHREADS)
    
    for k in range(Nsteps):
        print("step:"+str(k),end="\r")
        if LOGFILE!=None:
            log_file = open(LOGFILE,"a+")
            log_file.write("step:"+str(k)+"\n")
            log_file.close()               
        inputs = [(U_start,[],mps.gamma1,mps.svs[0],mps.gammas[0],mps.svs[1],d_max,cutoff)]
        for n in range(1,n2-1):
            #mps.apply_Ugate(U_even,2*n,d_max=d_max,cutoff=cutoff)
            r = 2*n
            inputs.append((U_even,mps.svs[r-1],mps.gammas[r-1],mps.svs[r],mps.gammas[r],mps.svs[r+1],d_max,cutoff))
       
        inputs.append((U_end,mps.svs[n_sites-3],mps.gammas[n_sites-3],mps.svs[n_sites-2],mps.gammaN,[],d_max,cutoff))
        
        print("step:"+str(k)+" start even process-map",end="\r")
        results = mergelistN(pool_N.map(apply_Ugates,splitlistN(inputs,NUMTHREADS)),NUMTHREADS)
        print("done",end="\r")
        
        #mps.apply_Ugate(U_end,2*(n2-1),d_max=d_max,cutoff=cutoff)
        gammal,sN,gammaN = results[-1]
        #mps.apply_Ugate(U_start,0,d_max=d_max,cutoff=cutoff)
        gamma1,s0,gammar = results[0]
        mps.gamma1 = gamma1
        mps.svs[0] = s0
        mps.gammas[0] = gammar
        mps.gammas[-1] = gammal
        mps.svs[-1] = sN
        mps.gammaN = gammaN
        
        for n in range(1,n2-1):
            r = 2*n
            gammal,s,gammar = results[n]
            mps.svs[r] = s
            mps.gammas[r-1]=gammal
            mps.gammas[r]=gammar
            
        inputs = []
        for n in range(0,no):
            #print("apply U_odd to "+str(2*n+1))
            #mps.apply_Ugate(U_odd,2*n+1,d_max=d_max,cutoff=cutoff)
            r = 2*n+1
            inputs.append((U_odd,mps.svs[r-1],mps.gammas[r-1],mps.svs[r],mps.gammas[r],mps.svs[r+1],d_max,cutoff))
            
        print("step:"+str(k)+" start odd process-map",end="\r")
        results = mergelistN(pool_N.map(apply_Ugates,splitlistN(inputs,NUMTHREADS)),NUMTHREADS)
        print("done",end="\r")
        for n in range(0,no):
            r = 2*n+1
            gammal,s,gammar = results[n]
            mps.svs[r] = s
            mps.gammas[r-1]=gammal
            mps.gammas[r]=gammar
          
        print("step:"+str(k)+" computing IAB               ",end="\r")
        data.append(func(mps))
        print("step:"+str(k)+" done                                   ",end="\r")
        print(str(mps.get_shape()))
    
    pool_N.close()
    return mps, data
    
    
def Ising_TEBD_symm_edge(mps0,delta_t,Nsteps,h,g,J,func=None,d_max=-1,cutoff=(10**(-16)),NUMTHREADS=4,LOGFILE=None): #symm + add edge J-terms
    h_bar = 1
    coupling = h_bar*h_bar*J#/4.0
    gfield = h_bar*g#/2.0
    hfield = h_bar*h#/2.0
    
    print("(h,g,J)="+str((h,g,J)))
    
    H_even = np.array([[hfield+coupling,0,gfield,0],[0,hfield-coupling,0,gfield],[gfield,0,-hfield-coupling,0],[0,gfield,0,coupling-hfield]])
    d,W = np.linalg.eigh(H_even)
    D = np.diag(np.exp(-1j/h_bar*delta_t*d))
    U_even = np.matmul(W,np.matmul(D,np.transpose(np.conjugate(W)))).reshape((2,2,2,2))
    
    U_odd = U_even
    
    H_start = np.array([[hfield,0,gfield,0],[0,hfield-2*coupling,0,gfield],[gfield,0,-hfield,0],[0,gfield,0,2*coupling-hfield]])
    d,W = np.linalg.eigh(H_start)
    D = np.diag(np.exp(-1j/h_bar*delta_t*d))
    U_start = np.matmul(W,np.matmul(D,np.transpose(np.conjugate(W)))).reshape((2,2,2,2))
    
    #H_end= np.array([[hfield,0,gfield,0],[0,hfield,0,gfield],[gfield,0,-hfield-2*coupling,0],[0,gfield,0,2*coupling-hfield]])
    H_end = np.array([[2*hfield,gfield,gfield,0],[gfield,0,0,gfield],[gfield,0,-2*coupling,gfield],[0,gfield,gfield,2*coupling-2*hfield]])
    d,W = np.linalg.eigh(H_end)
    D = np.diag(np.exp(-1j/h_bar*delta_t*d))
    U_end = np.matmul(W,np.matmul(D,np.transpose(np.conjugate(W)))).reshape((2,2,2,2))
    
    mps = mps0.copy()
    n_sites = len(mps.gammas)+2
    n2 = n_sites//2
    no = n2
    if n_sites%2==0:
       no -= 1
    
    if func==None:
        func = lambda x : x.get_EE(n2) 
        
    print(str((n2,no)))
        
    data = [func(mps)]
    print(str(mps.shape))
    print("data="+str(data[-1]))
    
    if NUMTHREADS<1:
        for k in range(Nsteps):
            print("step:"+str(k),end="\r")
            if LOGFILE!=None:
                log_file = open(LOGFILE,"a+")
                log_file.write("step:"+str(k)+"\n")
                log_file.close() 
            for n in range(1,n2-1):
                mps.apply_Ugate(U_even,2*n,d_max=d_max,cutoff=cutoff)
            mps.apply_Ugate(U_end,2*(n2-1),d_max=d_max,cutoff=cutoff)
            mps.apply_Ugate(U_start,0,d_max=d_max,cutoff=cutoff)

            for n in range(0,no):
                mps.apply_Ugate(U_odd,2*n+1,d_max=d_max,cutoff=cutoff)  
            data.append(func(mps))
        return mps,data
    
    pool_N = Pool(processes=NUMTHREADS)
    
    for k in range(Nsteps):
        print("step:"+str(k),end="\r")
        if LOGFILE!=None:
            log_file = open(LOGFILE,"a+")
            log_file.write("step:"+str(k)+"\n")
            log_file.close() 
        inputs = [(U_start,[],mps.gamma1,mps.svs[0],mps.gammas[0],mps.svs[1],d_max,cutoff)]
        for n in range(1,n2-1):
            #mps.apply_Ugate(U_even,2*n,d_max=d_max,cutoff=cutoff)
            r = 2*n
            inputs.append((U_even,mps.svs[r-1],mps.gammas[r-1],mps.svs[r],mps.gammas[r],mps.svs[r+1],d_max,cutoff))
       
        inputs.append((U_end,mps.svs[n_sites-3],mps.gammas[n_sites-3],mps.svs[n_sites-2],mps.gammaN,[],d_max,cutoff))
        
        print("step:"+str(k)+" Start even process-map",end="\r")
        results = mergelistN(pool_N.map(apply_Ugates,splitlistN(inputs,NUMTHREADS)),NUMTHREADS)
        print("done",end="\r")
        
        #mps.apply_Ugate(U_end,2*(n2-1),d_max=d_max,cutoff=cutoff)
        gammal,sN,gammaN = results[-1]
        #mps.apply_Ugate(U_start,0,d_max=d_max,cutoff=cutoff)
        gamma1,s0,gammar = results[0]
        mps.gamma1 = gamma1
        mps.svs[0] = s0
        mps.gammas[0] = gammar
        mps.gammas[-1] = gammal
        mps.svs[-1] = sN
        mps.gammaN = gammaN
        
        for n in range(1,n2-1):
            r = 2*n
            gammal,s,gammar = results[n]
            mps.svs[r] = s
            mps.gammas[r-1]=gammal
            mps.gammas[r]=gammar
            
        inputs = []
        for n in range(0,no):
            #print("apply U_odd to "+str(2*n+1))
            #mps.apply_Ugate(U_odd,2*n+1,d_max=d_max,cutoff=cutoff)
            r = 2*n+1
            inputs.append((U_odd,mps.svs[r-1],mps.gammas[r-1],mps.svs[r],mps.gammas[r],mps.svs[r+1],d_max,cutoff))
            
        print("step:"+str(k)+" start odd process-map",end="\r")
        results = mergelistN(pool_N.map(apply_Ugates,splitlistN(inputs,NUMTHREADS)),NUMTHREADS)
        print("done",end="\r")
        for n in range(0,no):
            r = 2*n+1
            gammal,s,gammar = results[n]
            mps.svs[r] = s
            mps.gammas[r-1]=gammal
            mps.gammas[r]=gammar
          
        data.append(func(mps))
        print("step done")
    
    pool_N.close()
    return mps, data
    

def test_TEBD(mps0,t_step,Nsteps,h,g,J,func=None,algorithms=[],d_max=[],cutoff=[],NUMTHREADS=4,LOGFILE="TEBD_logs.txt",verbose=False):
    n_tests = len(h)       
    if algorithms==[]:
        algorithms = [(-1,)]*n_tests
        
    if d_max==[]:
        d_max = [-1]*((algorithm==0)*1+1)*n_tests
    if cutoff==[]:
        cutoff = [(10**(-16))]*((algorithm==0)*1+1)*n_tests
    
    tebd_logfile = None
    if verbose:
        tebd_logfile = LOGFILE
        
    mpsf = []
    data = []
    for k in range(n_tests):
        if (-1 in algorithms[k]):
            t0 = time.time()
            ss = "TEBD(asymm+edge) NUMTHREADS="+str(NUMTHREADS)+" simulating (h,g,J)="+str((h[k],g[k],J[k]))+",t_step="+str(t_step[k])+",N="+str(Nsteps[k])+",d_max="+str(d_max[k])+",cutoff="+str(cutoff[k])
            print(ss)
            log_file = open(LOGFILE,"a+")
            log_file.write(ss+"\n")
            log_file.close()
            mpsfk,datak =Ising_TEBD_asymm_edge(mps0[k],t_step[k],Nsteps[k],h[k],g[k],J[k],func=func,d_max=d_max[k],cutoff=cutoff[k],NUMTHREADS=NUMTHREADS,LOGFILE=tebd_logfile)
            t1 = time.time()
            tstring = "Time-taken="+str(t1-t0)+"s"
            print(tstring)
            log_file = open(LOGFILE,"a+")
            log_file.write(tstring+"\n")
            log_file.close()             
            mpsf.append(mpsfk)
            data.append(datak)
        if (1 in algorithms[k]):
            t0 = time.time()
            ss = "TEBD(symm+edge) NUMTHREADS="+str(NUMTHREADS)+" simulating (h,g,J)="+str((h[k],g[k],J[k]))+",t_step="+str(t_step[k])+",N="+str(Nsteps[k])+",d_max="+str(d_max[k])+",cutoff="+str(cutoff[k])
            print(ss)
            log_file = open(LOGFILE,"a+")
            log_file.write(ss+"\n")
            log_file.close()
            mpsfk,datak =Ising_TEBD_symm_edge(mps0[k],t_step[k],Nsteps[k],h[k],g[k],J[k],func=func,d_max=d_max[k],cutoff=cutoff[k],NUMTHREADS=NUMTHREADS,LOGFILE=tebd_logfile)
            t1 = time.time()
            tstring = "Time-taken="+str(t1-t0)+"s"
            print(tstring)
            log_file = open(LOGFILE,"a+")
            log_file.write(tstring+"\n")
            log_file.close()             
            mpsf.append(mpsfk)
            data.append(datak)
        if (0 in algorithms[k]):
            t0 = time.time()
            ss = "FullH simulating (h,g,J)="+str((h[k],g[k],J[k]))+",t_step="+str(t_step[k])+",N="+str(Nsteps[k])+",d_max="+str(d_max[k])+",cutoff="+str(cutoff[k])
            print(ss)
            log_file = open(LOGFILE,"a+")
            log_file.write(ss+"\n")
            log_file.close()
            mpsfk,datak =Ising_fullH_MPS(mps0[k],t_step[k],Nsteps[k],h[k],g[k],J[k],func=func,d_max=d_max[k],cutoff=cutoff[k])
            t1 = time.time()
            tstring = "Time-taken="+str(t1-t0)+"s"
            print(tstring)
            log_file = open(LOGFILE,"a+")
            log_file.write(tstring+"\n")
            log_file.close()             
            mpsf.append(mpsfk)
            data.append(datak)
            
    
                               
    return mpsf,data

def test_fullH(S0,t_step,Nsteps,h,g,J,d_max=[],cutoff=[],LOGFILE="FullH_logs.txt"):
    n_tests = len(h)       
    if d_max==[]:
        d_max = [-1]*n_tests
    if cutoff==[]:
        cutoff = [(10**(-16))]*n_tests
    
    Sf = []
    data = []
    
    for k in range(n_tests):
        t0 = time.time()
        ss = "FullH simulating (h,g,J)="+str((h[k],g[k],J[k]))+",t_step="+str(t_step[k])+",N="+str(Nsteps[k])+",d_max="+str(d_max[k])+",cutoff="+str(cutoff[k])
        print(ss)
        log_file = open(LOGFILE,"a+")
        log_file.write(ss+"\n")
        log_file.close()
        Sfk,datak =Ising_fullH_EE(S0[k],t_step[k],Nsteps[k],h[k],g[k],J[k],d_max=d_max[k],cutoff=cutoff[k])
        t1 = time.time()
        tstring = "Time-taken="+str(t1-t0)+"s"
        print(tstring)
        log_file = open(LOGFILE,"a+")
        log_file.write(tstring+"\n")
        log_file.close()
        Sf.append(Sfk)
        data.append(datak)
      
    return Sf,data
    
def TEBD_randomBloch(L,N,t_step=0.01,Nsteps=2000,h=(5**0.5+1)/4.0,g=(5**0.5+5)/8.0,J=1,d_max=-1,cutoff=(10**(-16)),NUMTHREADS=4,LOGFILE="TEBD_RB_logs.txt",THETA=[],PHI=[]):
    func_EE = lambda x : x.get_EE(L//2)
    mpsf = []
    data = []
    theta = THETA
    phi = PHI
    if THETA==[]:
        log_file = open(LOGFILE,"a+")
        log_file.write("generating "+str(N)+" random theta\n")
        log_file.close()
        theta = [np.random.random_sample((L,))*np.pi for k in range(N)]
    if PHI==[]:
        log_file = open(LOGFILE,"a+")
        log_file.write("generating "+str(N)+" random phi\n")
        log_file.close()
        phi = [np.random.random_sample((L,))*np.pi*2 for k in range(N)]
        
    for k in range(N):
        #generate random state on Bloch-sphere
        thetak = theta[k] #
        phik = phi[k] #
        print("cMPS(theta="+str(thetak)+",phi="+str(phik)+")")
        mps0 = cMPS(state_from_polar(thetak,phik))
        print(len(mps0.svs))
        ss = str(k)+":TEBD(asymm+edge) simulating (theta="+str(thetak)+",phi="+str(phik)+")"
        print(ss)
        log_file = open(LOGFILE,"a+")
        log_file.write(ss+"\n")
        log_file.close()
        mpsfk,datak =Ising_TEBD_asymm_edge(mps0,t_step,Nsteps,h,g,J,func=func_EE,d_max=d_max,cutoff=cutoff,NUMTHREADS=NUMTHREADS)
        mpsf.append(mpsfk)
        data.append(datak)
        theta.append(thetak)
        phi.append(phik)
        
    return mpsf,data,theta,phi
  
def state_from_polar(thetas,phis):
    S = np.array([np.cos(thetas[0]/2),np.exp(1j*phis[0])*np.sin(thetas[0]/2)])
    for k in range(1,len(thetas)):
        S = np.kron(S,np.array([np.cos(thetas[k]/2),np.exp(1j*phis[k])*np.sin(thetas[k]/2)]))
    return S.reshape(tuple([2 for d in range(len(thetas))]))
    
def generate_NeelMPS(L):
    oneup = np.array([[[1.0],[0.0]]])
    onedown = np.array([[[0.0],[1.0]]])
    gammas = []
    for k in range(L//2-1):
        gammas.append(onedown)
        gammas.append(oneup)
        
    gammas[L//2-1]=-gammas[L//2-1]
    return MPS([np.array([1.0])]*(L-1),np.array([[1],[0]]),gammas,np.array([[0.0],[-1.0]]))
        

    
def print_help():
    print("arguments:")
    print("  --help                    show this help")
    print("  -v, --verbose                 write all steps to logs")
    print("arguments with input:")
    print("  -h, --zfield: float               value of field z-component (default: (5^0.5+1)/4 )")
    print("  -g, --xfield: float               value of field x-component (default: (5^0.5+8)/8 )")
    print("  -j, --coupling: float             value of spin coupling constant (default: 1 )")
    print("  -s, --start:  string              name of .pickle-file with intial MPS (default: Neel-state)")
    print("  -l, --length: integer             length of chain (default: 20)")
    print("  -n, --nsteps: integer             number of TEBD steps in simulation (default: 1000)")
    print("  -t, --tstep:  float               value of time-step in TEBD simulation (default: 0.01 )")
    print("  -d, --dmax:  float                maximal bond dimension of MPS (default: -1 (no truncation) )")
    print("  -c, --cutoff:  float              minimal schmidt-coefficient value of MPS (default: 10^(-16))")
    print("  -i, --input:  string              csv-file with multiple values for some/all arguments (h,g,j,s,l,n,t,d,c,i,a,v)")
    print("  -o, --output:  string             output .pickle-files")
    print("  -a , --algorithm: asymm symm      TEBD algorithm to use")
    print("  --threads:  integer               number of threads (default: 0 (no multithreading))")
    print("  --logs:  string                   filename for logs (default: TEBD_logs.txt)")
    print("  --comment:  string                comment at start of computation in logs")
    
 
class TEBD_input_args:
    def __init__(self,L=None,mps0=None,t_step=None,Nsteps=None,h=None,g=None,J=None,func=None,algorithm=None,d_max=None,cutoff=None,NUMTHREADS=None,LOGFILE=None,verbose=False):
    #mps0,t_step,Nsteps,h,g,J,func=None,algorithm=0,d_max=[],cutoff=[],NUMTHREADS=4,LOGFILE="TEBD_logs.txt"
    #mps0,delta_t,Nsteps,h,g,J,func=None,d_max=-1,cutoff=(10**(-16)),NUMTHREADS=4,LOGFILE=None
        self.L = L
        self.mps0 = mps0
        if mps0!=None:
            self.L = len(mps0.gammas)+2
        self.t_step = t_step
        self.Nsteps = Nsteps
        self.h = h
        self.g = g
        self.J = J
        self.func = None
        self.algorithm = algorithm
        self.d_max = d_max
        self.cutoff = cutoff
        self.NUMTHREADS = NUMTHREADS
        self.LOGFILE = LOGFILE
        self.verbose = verbose
        
    def copy(self):
        return TEBD_input_args(L=self.L,mps0=self.mps0,t_step=self.t_step,Nsteps=self.Nsteps,h=self.h,g=self.g,J=self.J,func=self.func,algorithm=self.algorithm,d_max=self.d_max,cutoff=self.cutoff,NUMTHREADS=self.NUMTHREADS,LOGFILE=self.LOGFILE,verbose=self.verbose)
        
    def merge(self,other):
        res = self.copy()
        if self.L==None:
            res.L = other.L
        if self.mps0==None and (self.L==other.L or self.L==None):
            res.mps0 = other.mps0
        if self.t_step==None:
            res.t_step = other.t_step
        if self.Nsteps==None:
            res.Nsteps = other.Nsteps
        if self.h==None:
            res.h = other.h
        if self.g==None:
            res.g = other.g
        if self.J==None:
            res.J = other.J
        if self.func==None:
            res.func = lambda x : (x.get_IAB(2,res.L-1), x.get_EE(res.L//2))
        if self.algorithm==None:
            res.algorithm = other.algorithm
        if self.d_max==None:
            res.d_max = other.d_max
        if self.cutoff==None:
            res.cutoff = other.cutoff
        if self.NUMTHREADS==None:
            res.NUMTHREADS = other.NUMTHREADS
        if self.LOGFILE==None:
            res.LOGFILE = other.LOGFILE
        if self.verbose or other.verbose:
            res.verbose = True
        return res
        
    def reset_default(self):
        self.L = 20
        self.mps0 = generate_NeelMPS(L)
        self.t_step = 0.01
        self.Nsteps = 1000
        self.h = (5**0.5+1)/4.0
        self.g = (5**0.5+5)/8.0
        self.J = 1
        self.func = lambda x : (x.get_IAB(2,self.L-1), x.get_EE(self.L//2))
        self.algorithm = 0
        self.d_max = -1
        self.cutoff = 10**(-16)
        self.NUMTHREADS = 4
        self.LOGFILE = "TEBD_logs.txt"
        self.verbose = False
        
    def L2NeelMPS(self):
        if self.L!=None and self.mps0==None:
            self.mps0 = generate_NeelMPS(self.L)
            
    def L2func(self):
        if self.L!=None:
            self.func = lambda x : (x.get_IAB(2,self.L-1), x.get_EE(self.L//2))
        
    def __str__(self,mps2str=False):
        mps0str = ", mps0=None"
        if self.mps0!=None:
            if mps2str:
                mps0str = ", mps0="+str(self.mps0)
            else:
                mps0str = ", mps0.shape="+str(self.mps0.shape)
                
        ss = "L="+str(self.L) + mps0str+ ", t_step="+str(self.t_step)+", Nsteps="+str(self.Nsteps)+", (h,g,J)="+str((self.h,self.g,self.J))+" func="+str(self.func)+", algorithm="+str(self.algorithm)+", d_max="+str(self.d_max)+", cutoff="+str(self.cutoff)+", NUMTHREADS="+str(self.NUMTHREADS)+", LOGFILE="+str(self.LOGFILE)+", verbose="+str(self.verbose)
        return ss
 
def opts2argclass(opts):#h,g,j,s,l,n,t,i,d,c,a,v--logs,--threads
    argclass = TEBD_input_args()
    for o,a in opts:
        if o in ("-h","--zfield"):
            argclass.h=float(a)
        elif o in ("-g","--xfield"):
            argclass.g=float(a)
        elif o in ("-j","--coupling"):
            argclass.J=float(a)
        elif o in ("-s", "--start"):
            argclass.mps0_file = a
        elif o in ("-l","--length"):
            argclass.L = int(a)
        elif o in ("-n","--nsteps"):
            argclass.Nsteps = int(a)
        elif o in ("-t","--tstep"):
            argclass.t_step=float(a)
        elif o in ("-i","--input"):
            argclass.csvinput = a
        elif o in ("-d", "--dmax"):
            argclass.d_max = int(a)
        elif o in ("-c", "--cutoff"):
            argclass.cutoff = float(a)
        elif o=="--logs":
            argclass.LOG_FILE = a
        elif o=="--threads":
            argclass.NUMTHREADS = int(a)
        elif o in ("-a","--algorithm"):
            argclass.algorithm = a
        elif o in ("-v","--verbose"):
            argclass.verbose = True
        else:
            assert False , "Unrecognised option "+str((o,a))
    return argclass
 
def parse_input_csv(file_name,DICT={}):
    f = open(file_name,"r")
    lines = [line.rstrip('\n').split(',') for line in f.readlines()]
    names = lines[0]
    TEBDargs = []
    for line in lines[1:]:
        opts = []
        for k in range(len(line)):
            val = line[k].replace(' ','')
            if val!='':
                o = names[k]
                a = val
                if a in DICT:
                    a = DICT[a]
                opts.append((o,a))
        print(opts)        
        TEBDargs.append(opts2argclass(opts))
    return TEBDargs

def main(argv):
    print(str(argv))
    
    LOG_FILE = "TEBD_logs_IAB.txt"
    NUM_THREADS = 4
    DICT = {
    "asymm":-1,
    "symm":1,
    "both":0
    }
    h = (5**0.5+1)/4.0
    g = (5**0.5+5)/8.0
    J = 1
    t_step = 0.02
    N = 500
    L = 8
    d_max = -1
    cutoff = 10**(-16)
    
    csvinput = None
    mps0_file = None
    algorithm = -1
    verbose_log = False
    comment = None
    pname = "testIAB"
    nparams = 1
     
    try:
        opts,args = getopt.getopt(argv,"h:g:j:l:n:t:s:i:a:o:d:c:v",["help","zfield=","xfield=","coupling=","start=","input=","length=","nsteps=","tstep=","logs=","algorithm=","output=","dmax=","cutoff=","comment=","threads=","verbose"])
        print(str(opts))
        print(str(args))
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)
    
    for o,a in opts:
        if o=="--help":
            print_help()
            sys.exit()
        elif o in ("-h","--zfield"):
            h=float(a)
        elif o in ("-g","--xfield"):
            g=float(a)
        elif o in ("-j","--coupling"):
            J=float(a)
        elif o in ("-s", "--start"):
            mps0_file = a
        elif o in ("-l","--length"):
            L = int(a)
        elif o in ("-n","--nsteps"):
            N = int(a)
        elif o in ("-t","--tstep"):
            t_step=float(a)
        elif o in ("-i","--input"):
            csvinput = a
        elif o in ("-d", "--dmax"):
            d_max = int(a)
        elif o in ("-c", "--cutoff"):
            cutoff = float(a)
        elif o=="--logs":
            LOG_FILE = a
        elif o=="--threads":
            NUM_THREADS = int(a)
        elif o in ("-a","--algorithm"):
            algorithm = DICT[a]
        elif o in ("-v","--verbose"):
            verbose_log = True
        elif o in ("-o","--output"):
            pname = a
        elif o=="--comment":
            comment = a
        else:
            assert False , "Error unrecognised option \""+str(o)+"\""
     
    #print(str((opts2argclass(opts))))     
    command_options = TEBD_input_args(L=L,mps0=None,t_step=t_step,Nsteps=N,h=h,g=g,J=J,func=None,algorithm=algorithm,d_max=d_max,cutoff=cutoff,NUMTHREADS=NUM_THREADS,LOGFILE=LOG_FILE,verbose=verbose_log)
    print(str(command_options))
    
    
    mps0 = None
    if mps0_file==None:
        mps0 =  [generate_NeelMPS(L)]
    else:
        mps0 = pickle.load(open(mps0_file,"rb"))
        print("Loading starting state(s) from "+str(mps0_file))
    
    options = []
    if csvinput!=None:
        csv_options = parse_input_csv(csvinput,DICT=DICT)
        for opt in csv_options:
            print(str(opt))
        nparams = len(csv_options)
        for k in range(nparams):
            command_options.mps0 = mps0[k%len(mps0)]
            opt = csv_options[k]
            res = opt.merge(command_options)
            res.L2func()
            res.L2NeelMPS()
            print(str(res))
            options.append(res)
    else:
        nparams = len(mps0)
        for m in mps0:
            opt = TEBD_input_args(mps0=m)
            options.append(opt.merge(command_options))
        
    if comment==None:
        comment = "Starting TEBD run with N="+str(N)+" steps for L="+str(L)+" state\n"
        
    f = open(LOG_FILE,"a+")
    f.write(comment)
    f.close()
    
    start_time0 = time.time()
    
    print(nparams)
    print(len(mps0))
    
    # mpsf,data = test_TEBD(mps0,[t_step]*nparams,[N]*nparams,[h]*nparams,[g]*nparams,[J]*nparams,func=None,algorithm=-1,d_max=d_maxs,cutoff=cutoffs,NUMTHREADS=NUM_THREADS,LOGFILE=LOG_FILE,verbose=verbose_log)
       
    mpsf = []
    data = []
    for k in range(nparams):
        opts = options[k]
        print(str(opts))
        tebd_logfile = None
        if opts.verbose:
            tebd_logfile = opts.LOGFILE
        if opts.algorithm==-1:
            t0 = time.time()
            ss = "TEBD(asymm+edge) NUMTHREADS="+str(opts.NUMTHREADS)+" simulating L="+str(opts.L)+" state with (h,g,J)="+str((opts.h,opts.g,opts.J))+",t_step="+str(opts.t_step)+",N="+str(opts.Nsteps)+",d_max="+str(opts.d_max)+",cutoff="+str(opts.cutoff)
            print(ss)
            log_file = open(opts.LOGFILE,"a+")
            log_file.write(ss+"\n")
            log_file.close()
            mpsfk,datak =Ising_TEBD_asymm_edge(opts.mps0,opts.t_step,opts.Nsteps,opts.h,opts.g,opts.J,func=opts.func,d_max=opts.d_max,cutoff=opts.cutoff,NUMTHREADS=opts.NUMTHREADS,LOGFILE=tebd_logfile)
            t1 = time.time()
            tstring = "Time-taken="+str(t1-t0)+"s"
            print(tstring)
            log_file = open(opts.LOGFILE,"a+")
            log_file.write(tstring+"\n")
            log_file.close()             
            mpsf.append(mpsfk)
            data.append(datak)
        if  opts.algorithm==1:
            t0 = time.time()
            ss = "TEBD(symm+edge) NUMTHREADS="+str(opts.NUMTHREADS)+" simulating L="+str(opts.L)+" state with (h,g,J)="+str((opts.h,opts.g,opts.J))+",t_step="+str(opts.t_step)+",N="+str(opts.Nsteps)+",d_max="+str(opts.d_max)+",cutoff="+str(opts.cutoff)
            print(ss)
            log_file = open(opts.LOGFILE,"a+")
            log_file.write(ss+"\n")
            log_file.close()
            mpsfk,datak =Ising_TEBD_symm_edge(opts.mps0,opts.t_step,opts.Nsteps,opts.h,opts.g,opts.J,func=opts.func,d_max=opts.d_max,cutoff=opts.cutoff,NUMTHREADS=opts.NUMTHREADS,LOGFILE=tebd_logfile)
            t1 = time.time()
            tstring = "Time-taken="+str(t1-t0)+"s"
            print(tstring)
            log_file = open(opts.LOGFILE,"a+")
            log_file.write(tstring+"\n")
            log_file.close()             
            mpsf.append(mpsfk)
            data.append(datak)
        if opts.algorithm==0:
            t0 = time.time()
            ss = "FullH simulating (h,g,J)="+str((opts.h,opts.g,opts.J))+",t_step="+str(opts.t_step)+",N="+str(opts.Nsteps)+",d_max="+str(opts.d_max)+",cutoff="+str(opts.cutoff)
            print(ss)
            log_file = open(opts.LOGFILE,"a+")
            log_file.write(ss+"\n")
            log_file.close()
            mpsfk,datak =Ising_fullH_MPS(opts.mps0,opts.t_step,opts.Nsteps,opts.h,opts.g,opts.J,func=opts.func,d_max=opts.d_max,cutoff=opts.cutoff)
            t1 = time.time()
            tstring = "Time-taken="+str(t1-t0)+"s"
            print(tstring)
            log_file = open(opts.LOGFILE,"a+")
            log_file.write(tstring+"\n")
            log_file.close()             
            mpsf.append(mpsfk)
            data.append(datak)
            

    end_time0 = time.time()
    print("TOTAL TIME="+str(end_time0-start_time0)+"s")
    f = open(LOG_FILE,"a+")
    f.write("TOTAL TIME="+str(end_time0-start_time0)+"s\n")
    f.write("Writing results to results/..."+pname+".pickle\n\n")
    f.close()
    print("Writing results to results/..."+pname+".pickle")

    pickle.dump(data,open("results/data"+pname+".pickle","wb"))
    pickle.dump(mpsf,open("results/mpsf"+pname+".pickle","wb"))
    return 0

if __name__=='__main__':
    main(sys.argv[1:])
    