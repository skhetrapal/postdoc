#define M "4"
#define P "5"
#define Hamiltonian "(ab(0)*ab(0)*a(0)*a(0)*C0C0C0C0 + 4*ab(0)*ab(1)*a(0)*a(1)*C0C1C0C1 + 4*ab(0)*ab(2)*a(0)*a(2)*C0C2C0C2 + 2*ab(0)*ab(2)*a(1)*a(1)*C0C2C1C1 + 4*ab(0)*ab(3)*a(0)*a(3)*C0C3C0C3 + 4*ab(0)*ab(3)*a(1)*a(2)*C0C3C1C2 + 
4*ab(0)*ab(4)*a(0)*a(4)*C0C4C0C4 + 4*ab(0)*ab(4)*a(1)*a(3)*C0C4C1C3 + 
2*ab(0)*ab(4)*a(2)*a(2)*C0C4C2C2 + 2*ab(1)*ab(1)*a(0)*a(2)*C0C2C1C1 +    ab(1)*ab(1)*a(1)*a(1)*C1C1C1C1 + 4*ab(1)*ab(2)*a(0)*a(3)*C0C3C1C2 +   4*ab(1)*ab(2)*a(1)*a(2)*C1C2C1C2 + 4*ab(1)*ab(3)*a(0)*a(4)*C0C4C1C3 + 4*ab(1)*ab(3)*a(1)*a(3)*C1C3C1C3 + 2*ab(1)*ab(3)*a(2)*a(2)*C1C3C2C2 + 2*ab(2)*ab(2)*a(0)*a(4)*C0C4C2C2 + 2*ab(2)*ab(2)*a(1)*a(3)*C1C3C2C2 + ab(2)*ab(2)*a(2)*a(2)*C2C2C2C2)"

#define v1 "sqrtinv(2)*a(0)*a(1)^2*a(2)"
#define v2 "sqrtinv(2)*a(0)^2*a(1)*a(3)"
#define v3 "sqrtinv(4)*a(0)^2*a(2)^2"
#define v4 "sqrtinv(6)*a(0)^3*a(4)"
#define v5 "sqrtinv(24)*a(1)^4"

#define vb1 "sqrtinv(2)*ab(0)*ab(1)^2*ab(2)"
#define vb2 "sqrtinv(2)*ab(0)^2*ab(1)*ab(3)"
#define vb3 "sqrtinv(4)*ab(0)^2*ab(2)^2"
#define vb4 "sqrtinv(6)*ab(0)^3*ab(4)"
#define vb5 "sqrtinv(24)*ab(1)^4"

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

Symbols l,k,b,d;
CFunction sqrtinv;
Functions a,ab;

#do i=1,`P'
 #do j=`i', `P'
  Local H`i'H`j' = `v`i''*`Hamiltonian'*`vb`j'';
 #enddo
#enddo
;

#do m=0,`M'
#define Mh "`m'/2"
 #do i=0, `Mh'
  #do j=0, `Mh'
   id C`i'C{`m'-`i'}C`j'C{`m'-`j'} = 1/('m'+1);
  #enddo
 #enddo
#enddo
;

repeat;
id a(l?)*ab(k?)=delta_(l,k)+ab(k)*a(l);
endrepeat;

repeat;
id sqrtinv(d?)*sqrtinv(b?)=sqrtinv(d*b);
endrepeat;

id a(l?)=0;
id ab(k?)=0;

print;

.sort

#write <Write_nm1_4_4.txt> "[[%E,%E,%E,%E,%E],[%E,%E,%E,%E,%E],[%E,%E,%E,%E,%E],[%E,%E,%E,%E,%E],[%E,%E,%E,%E,%E]]" H1H1,H1H2,H1H3,H1H4,H1H5,H1H2,H2H2,H2H3,H2H4,H2H5,H1H3,H2H3,H3H3,H3H4,H3H5,H1H4,H2H4,H3H4,H4H4,H4H5,H1H5,H2H5,H3H5,H4H5,H5H5

.end
