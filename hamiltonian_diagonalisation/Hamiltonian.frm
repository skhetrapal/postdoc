#define M "9"
#define Mhalf "`M'/2"
    
Symbols m,n;
Functions a,ab;  
      
Symbols
#do i=0, `M'
 #do j=0, `M'
  C`i'C`j'
 #enddo
#enddo
;
        
Symbols
#do m=0,`M'
#define Mh "`m'/2"
 #do i=0, `Mh'
  #do j=0, `Mh'
   C`i'C{`m'-`i'}C`j'C{`m'-`j'}
  #enddo
 #enddo
#enddo
;
    
Functions HamiltonianL, HamiltonianR;
    
Local Hamiltonian = 
#do m=0, `M'
  + HamiltonianL(`m')*HamiltonianR(`m')
#enddo
;
    
#do m=0, `M'
    
#define Mh "`m'/2"
    
 id HamiltonianL(`m')=
  #do i=0, `Mh'
   #if `i'=={`m'-`i'}
    +C`i'C{`m'-`i'}*ab(`i')*ab({`m'-`i'})
   #else
     +2*C`i'C{`m'-`i'}*ab(`i')*ab({`m'-`i'})
   #endif
  #enddo
 ;
    
 id HamiltonianR(`m')=
  #do i=0, `Mh'
   #if `i'=={`m'-`i'}
    +C`i'C{`m'-`i'}*a(`i')*a({`m'-`i'})
   #else
    +2*C`i'C{`m'-`i'}*a(`i')*a({`m'-`i'})
   #endif
  #enddo
 ;
   
 #do i=0, `Mh'
  #do j=0, `Mh'
   id C`i'C{`m'-`i'}*C`j'C{`m'-`j'} = C`i'C{`m'-`i'}C`j'C{`m'-`j'};
  #enddo
 #enddo
 ;
    
 #enddo
    
print;
.end
