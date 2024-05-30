#define N "3"
#define M "10"

CFunction sqrtinv;
Symbols s,t,n,b,c,k;
Symbols
#do i=0,`M'
,a`i',par`i'
#enddo
;
Function a;

Local partitions=1
#do i=0,`M'
 *par`i'
#enddo
;

#do i=0,`M'
 id par`i'=1
 #do j=1,`N'
  #if {`j'*`i'}<=`M'
   +a`i'^`j'*s^`j'*t^(`i'*`j')*(sqrtinv(fac_('j')))
  #endif
 #enddo
;
id t^n?{>`M'}=0;
id s^n?{>`N'}=0;

.sort
#enddo

repeat;
id sqrtinv(b?)*sqrtinv(c?)=sqrtinv(b*c);
endrepeat;

#do i=0,`M'
 id a`i'=a(`i');
#enddo

id sqrtinv(1) = 1;

brackets t,s;
print +s;
.end
