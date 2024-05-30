#define M "8"
#define P "10"
#define Hamiltonian "(ab(0)*ab(0)*a(0)*a(0)*C0C0C0C0 + 4*ab(0)*ab(1)*a(0)*a(1)*C0C1C0C1 + 4*ab(0)*ab(2)*a(0)*a(2)*C0C2C0C2 + 2*ab(0)*ab(2)*a(1)*a(1)*C0C2C1C1 + 4*ab(0)*ab(3)*a(0)*a(3)*C0C3C0C3 + 4*ab(0)*ab(3)*a(1)*a(2)*C0C3C1C2 + 4*ab(0)*ab(4)*a(0)*a(4)*C0C4C0C4 + 4*ab(0)*ab(4)*a(1)*a(3)*C0C4C1C3 + 2*ab(0)*ab(4)*a(2)*a(2)*C0C4C2C2 + 4*ab(0)*ab(5)*a(0)*a(5)*C0C5C0C5 + 4*ab(0)*ab(5)*a(1)*a(4)*C0C5C1C4 + 4*ab(0)*ab(5)*a(2)*a(3)*C0C5C2C3 + 4*ab(0)*ab(6)*a(0)*a(6)*C0C6C0C6 + 4*ab(0)*ab(6)*a(1)*a(5)*C0C6C1C5 + 4*ab(0)*ab(6)*a(2)*a(4)*C0C6C2C4 + 2*ab(0)*ab(6)*a(3)*a(3)*C0C6C3C3 + 4*ab(0)*ab(7)*a(0)*a(7)*C0C7C0C7 + 4*ab(0)*ab(7)*a(1)*a(6)*C0C7C1C6 + 4*ab(0)*ab(7)*a(2)*a(5)*C0C7C2C5 + 4*ab(0)*ab(7)*a(3)*a(4)*C0C7C3C4 + 4*ab(0)*ab(8)*a(0)*a(8)*C0C8C0C8 + 4*ab(0)*ab(8)*a(1)*a(7)*C0C8C1C7 + 4*ab(0)*ab(8)*a(2)*a(6)*C0C8C2C6 + 4*ab(0)*ab(8)*a(3)*a(5)*C0C8C3C5 + 2*ab(0)*ab(8)*a(4)*a(4)*C0C8C4C4 + 2*ab(1)*ab(1)*a(0)*a(2)*C0C2C1C1 + ab(1)*ab(1)*a(1)*a(1)*C1C1C1C1 + 4*ab(1)*ab(2)*a(0)*a(3)*C0C3C1C2 + 4*ab(1)*ab(2)*a(1)*a(2)*C1C2C1C2 + 4*ab(1)*ab(3)*a(0)*a(4)*C0C4C1C3 + 4*ab(1)*ab(3)*a(1)*a(3)*C1C3C1C3 + 2*ab(1)*ab(3)*a(2)*a(2)*C1C3C2C2 + 4*ab(1)*ab(4)*a(0)*a(5)*C0C5C1C4 + 4*ab(1)*ab(4)*a(1)*a(4)*C1C4C1C4 + 4*ab(1)*ab(4)*a(2)*a(3)*C1C4C2C3 + 4*ab(1)*ab(5)*a(0)*a(6)*C0C6C1C5 + 4*ab(1)*ab(5)*a(1)*a(5)*C1C5C1C5 + 4*ab(1)*ab(5)*a(2)*a(4)*C1C5C2C4 + 2*ab(1)*ab(5)*a(3)*a(3)*C1C5C3C3 + 4*ab(1)*ab(6)*a(0)*a(7)*C0C7C1C6 + 4*ab(1)*ab(6)*a(1)*a(6)*C1C6C1C6 + 4*ab(1)*ab(6)*a(2)*a(5)*C1C6C2C5 + 4*ab(1)*ab(6)*a(3)*a(4)*C1C6C3C4 + 4*ab(1)*ab(7)*a(0)*a(8)*C0C8C1C7 + 4*ab(1)*ab(7)*a(1)*a(7)*C1C7C1C7 + 4*ab(1)*ab(7)*a(2)*a(6)*C1C7C2C6 + 4*ab(1)*ab(7)*a(3)*a(5)*C1C7C3C5 + 2*ab(1)*ab(7)*a(4)*a(4)*C1C7C4C4 + 2*ab(2)*ab(2)*a(0)*a(4)*C0C4C2C2 + 2*ab(2)*ab(2)*a(1)*a(3)*C1C3C2C2 + ab(2)*ab(2)*a(2)*a(2)*C2C2C2C2 + 4*ab(2)*ab(3)*a(0)*a(5)*C0C5C2C3 + 4*ab(2)*ab(3)*a(1)*a(4)*C1C4C2C3 + 4*ab(2)*ab(3)*a(2)*a(3)*C2C3C2C3 + 4*ab(2)*ab(4)*a(0)*a(6)*C0C6C2C4 + 4*ab(2)*ab(4)*a(1)*a(5)*C1C5C2C4 + 4*ab(2)*ab(4)*a(2)*a(4)*C2C4C2C4 + 2*ab(2)*ab(4)*a(3)*a(3)*C2C4C3C3 + 4*ab(2)*ab(5)*a(0)*a(7)*C0C7C2C5 + 4*ab(2)*ab(5)*a(1)*a(6)*C1C6C2C5 + 4*ab(2)*ab(5)*a(2)*a(5)*C2C5C2C5 + 4*ab(2)*ab(5)*a(3)*a(4)*C2C5C3C4 + 4*ab(2)*ab(6)*a(0)*a(8)*C0C8C2C6 + 4*ab(2)*ab(6)*a(1)*a(7)*C1C7C2C6 + 4*ab(2)*ab(6)*a(2)*a(6)*C2C6C2C6 + 4*ab(2)*ab(6)*a(3)*a(5)*C2C6C3C5 + 2*ab(2)*ab(6)*a(4)*a(4)*C2C6C4C4 + 2*ab(3)*ab(3)*a(0)*a(6)*C0C6C3C3 + 2*ab(3)*ab(3)*a(1)*a(5)*C1C5C3C3 + 2*ab(3)*ab(3)*a(2)*a(4)*C2C4C3C3 + ab(3)*ab(3)*a(3)*a(3)*C3C3C3C3 + 4*ab(3)*ab(4)*a(0)*a(7)*C0C7C3C4 + 4*ab(3)*ab(4)*a(1)*a(6)*C1C6C3C4 + 4*ab(3)*ab(4)*a(2)*a(5)*C2C5C3C4 + 4*ab(3)*ab(4)*a(3)*a(4)*C3C4C3C4 + 4*ab(3)*ab(5)*a(0)*a(8)*C0C8C3C5 + 4*ab(3)*ab(5)*a(1)*a(7)*C1C7C3C5 + 4*ab(3)*ab(5)*a(2)*a(6)*C2C6C3C5 + 4*ab(3)*ab(5)*a(3)*a(5)*C3C5C3C5 + 2*ab(3)*ab(5)*a(4)*a(4)*C3C5C4C4 + 2*ab(4)*ab(4)*a(0)*a(8)*C0C8C4C4 + 2*ab(4)*ab(4)*a(1)*a(7)*C1C7C4C4 + 2*ab(4)*ab(4)*a(2)*a(6)*C2C6C4C4 + 2*ab(4)*ab(4)*a(3)*a(5)*C3C5C4C4 + ab(4)*ab(4)*a(4)*a(4)*C4C4C4C4)"

#define v1 "sqrtinv(1)*a(1)*a(3)*a(4)"
#define v2 "sqrtinv(1)*a(1)*a(2)*a(5)"
#define v3 "sqrtinv(1)*a(0)*a(3)*a(5)"
#define v4 "sqrtinv(1)*a(0)*a(2)*a(6)"
#define v5 "sqrtinv(1)*a(0)*a(1)*a(7)"
#define v6 "sqrtinv(2)*a(2)*a(3)^2"
#define v7 "sqrtinv(2)*a(2)^2*a(4)"
#define v8 "sqrtinv(2)*a(1)^2*a(6)"
#define v9 "sqrtinv(2)*a(0)*a(4)^2"
#define v10 "sqrtinv(2)*a(0)^2*a(8)"

#define vb1 "sqrtinv(1)*ab(1)*ab(3)*ab(4)"
#define vb2 "sqrtinv(1)*ab(1)*ab(2)*ab(5)"
#define vb3 "sqrtinv(1)*ab(0)*ab(3)*ab(5)"
#define vb4 "sqrtinv(1)*ab(0)*ab(2)*ab(6)"
#define vb5 "sqrtinv(1)*ab(0)*ab(1)*ab(7)"
#define vb6 "sqrtinv(2)*ab(2)*ab(3)^2"
#define vb7 "sqrtinv(2)*ab(2)^2*ab(4)"
#define vb8 "sqrtinv(2)*ab(1)^2*ab(6)"
#define vb9 "sqrtinv(2)*ab(0)*ab(4)^2"
#define vb10 "sqrtinv(2)*ab(0)^2*ab(8)"

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
   id C`i'C{`m'-`i'}C`j'C{`m'-`j'} = 1/(`m'+1);
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
.end


