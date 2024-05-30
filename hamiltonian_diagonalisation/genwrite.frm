#define P "10"

#write <write.txt> "#write \"[[%%E%"
#do j=1,{`P'-1}
 #write <write.txt> ",%%E%"
#enddo
#write <write.txt> "]%"

#do i=1,{`P'-1}
#write <write.txt> ",[%%E%"
#do j=1,{`P'-1}
 #write <write.txt> ",%%E%"
#enddo
#write <write.txt> "]%"
#enddo

#write <write.txt> "]\"%"

#do i=1,`P'
#do j=1,`P'
 #if (`i'<`j')
  #write <write.txt> ",H`i'H`j'%"
 #else
  #write <write.txt> ",H`j'H`i'%"
 #endif
#enddo
#enddo


.end

