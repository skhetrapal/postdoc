#define P "10"

Functions
 #do i=1, `P'
  #do j=1, `P'
   H`i'H`j'
  #enddo
 #enddo
;

Local H = 1
 #do i=1, `P'
  #do j=1, `P'
    #if `i'<=`j'   
     *H`i'H`j'
    #else
     *H`j'H`i'
    #endif
  #enddo
 #enddo
 ;

print;
end.

