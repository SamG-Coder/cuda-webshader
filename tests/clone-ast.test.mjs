import test from 'node:test';
import assert from 'node:assert/strict';
import {cloneAst} from '../src/compiler/clone-ast.js';
test('AST clone preserves deep helper trees without recursive native cloning',()=>{const leaf={kind:'literal',value:7};let tree=leaf;for(let i=0;i<12000;i++)tree={kind:'binary',left:tree,right:leaf};const copy=cloneAst(tree);assert.notEqual(copy,tree);let p=copy;for(let i=0;i<12000;i++){assert.equal(p.kind,'binary');assert.equal(p.right,copy.right);p=p.left;}assert.equal(p,copy.right);assert.notEqual(p,leaf);});
test('AST clone preserves aliases, sparse arrays and cycles without mutating input',()=>{const a={token:{line:2},items:new Array(3)};a.items[2]=a.token;a.self=a;const b=cloneAst(a);assert.equal(b.self,b);assert.equal(b.items[2],b.token);assert.equal(0 in b.items,false);b.token.line=4;assert.equal(a.token.line,2);});
