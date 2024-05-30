#define P "5"

Functions
#do i=1,`P'
#do j=1,`P'
,H`i'H`j'
#enddo
#enddo
;

Local H=1
#do i=1,`P'
#do j=1,`P'
 *H`i'H`j'
#enddo
#enddo
;

#do i=1,`P'
#do j=1,`P'
 #if `j'<`i'
  id H`i'H`j'=H`j'H`i';
 #endif
#enddo
#enddo

print;
.end

