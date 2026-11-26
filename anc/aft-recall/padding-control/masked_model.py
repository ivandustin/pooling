import math
from flax import nnx
from jax import numpy as jnp

class Block(nnx.Module):
    def __init__(self, width, rngs, ff_hidden=None, ff_depth=1):
        ff_hidden=ff_hidden or 4*width
        self.n1 = nnx.LayerNorm(width, rngs=rngs)
        self.qkv = nnx.Linear(width, 3*width, rngs=rngs)
        self.proj = nnx.Linear(width, width, rngs=rngs)
        self.n2 = nnx.LayerNorm(width, rngs=rngs)
        self.up = nnx.Linear(width, ff_hidden, rngs=rngs)
        self.down = nnx.Linear(ff_hidden, width, rngs=rngs)
        self.ff_layers=nnx.List([nnx.Linear(ff_hidden,ff_hidden,rngs=rngs) for _ in range(ff_depth-1)])

    def __call__(self, x, valid_keys):
        b, length, width = x.shape
        qkv = self.qkv(self.n1(x)).reshape(b, length, 3, 4, width//4)
        q, k, v = [qkv[:,:,i].transpose(0,2,1,3) for i in range(3)]
        scores = q @ k.swapaxes(-1,-2) / math.sqrt(width//4)
        mask = jnp.arange(length)[:,None] >= jnp.arange(length)[None,:]
        mask = mask[None, None, :, :] & valid_keys[:, None, None, :]
        from jax.nn import softmax, gelu
        weights = softmax(jnp.where(mask, scores, -1e9), axis=-1)
        z = (weights @ v).transpose(0,2,1,3).reshape(b,length,width)
        x = x + self.proj(z)
        h=gelu(self.up(self.n2(x)))
        for layer in self.ff_layers:h=gelu(layer(h))
        return x+self.down(h)


class ProbeTransformer(nnx.Module):
    def __init__(self, vocab, context, width, rngs, depth=2, ff_hidden=None, ff_depth=1):
        self.token = nnx.Embed(vocab,width,rngs=rngs)
        self.position = nnx.Embed(context,width,rngs=rngs)
        self.blocks = nnx.List([Block(width,rngs,ff_hidden,ff_depth) for _ in range(depth)])
        self.norm = nnx.LayerNorm(width,rngs=rngs)
        self.head = nnx.Linear(width,vocab,rngs=rngs)

    def __call__(self, tokens, valid_keys=None):
        if valid_keys is None:
            valid_keys = tokens != 95
        x = self.token(tokens) + self.position(jnp.arange(tokens.shape[-1]))
        for block in self.blocks:
            x = block(x, valid_keys)
        return self.head(self.norm(x[:,-1]))

