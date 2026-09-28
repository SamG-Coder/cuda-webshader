export const examples = {
  pow2: {
    title:'Recursive fork / join', entry:'pow2', input:'[[0],[1],[2],[3],[4],[5],[6]]',
    source:`import Base

# Each branch is evaluated at runtime, including the recursive calls.
def pow2(+depth: Nat) -> U32:
  match depth:
    case 0n:
      1
    case 1n+p:
      a b = pow2(p) pow2(p)
      (a + b : U32)
`,
  },
  tree: {
    title:'Allocate and fold a tree', entry:'main', input:'[[0],[1],[2],[3],[4]]',
    source:`import Base

type Tree is Data:
  Leaf{value: U32}
  Branch{left: Tree, right: Tree}

def grow(+depth: Nat) -> Tree:
  match depth:
    case 0n:
      Leaf{3}
    case 1n+p:
      a b = grow(p) grow(p)
      Branch{a, b}

def total(tree: Tree) -> U32:
  match tree:
    case Leaf{value}:
      value
    case Branch{left, right}:
      a b = total(left) total(right)
      (a + b : U32)

def main(depth: Nat) -> U32:
  total(grow(depth))
`,
  },
  closure: {
    title:'Captured floating-point closure', entry:'main', input:'[[2,3],[0.5,4],[-2,8],[10,-0.25]]',
    source:`import Base

def offset(bias: F32) -> F32 -> F32:
  x => (x + bias : F32)

def main(x: F32, bias: F32) -> F32:
  offset(bias)(x)
`,
  },
  proof: {
    title:'Checked identity law', entry:'identity', input:'[[0],[1],[4294967295]]',
    source:`import Base

def identity(x: U32) -> U32:
  x

law unchanged:
  for x: U32
  {identity(x) == x : U32}

def unchanged(x):
  {==}
`,
  },
};
