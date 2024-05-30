#define N "10"
#define M "25"

CFunction ri;
Symbols s,t,n,b,c,k;
Symbols
#do i=0,`M'
,a`i',par`i'
#enddo
;
Function a;

Format 255;

Local partitions=1
#do i=0,`M'
 *par`i'
#enddo
;

#do i=0,`M'
 id par`i'=1
 #do j=1,`N'
  #if {`j'*`i'}<=`M'
   +a`i'^`j'*s^`j'*t^(`i'*`j')
    #do k=2,`j'
     *ri('k')
    #enddo
  #endif
 #enddo
;
id t^n?{>`M'}=0;
id s^n?{>`N'}=0;

.sort
#enddo

id t^n?{<`M'}=0;
id s^n?{<`N'}=0;

.sort

id t^n?{>`M'}=0;
id s^n?{>`N'}=0;

id t=1;
id s=1;

id ri(n?)^2=1/n;

print +s;
.end
