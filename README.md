Code for own implementation of:
Let's build GPT: from scratch, in code, spelled out. (Andrej Karpathy)
- https://youtu.be/kCc8FmEb1nY?si=cuKfZXjzhH1hNuCr&t=6044
- https://github.com/karpathy/ng-video-lecture

```
Using device: cuda
Dataset length: 1115394 characters
First 100 characters:
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You
Vocabulary size: 65
Train data length: 1003854
Val data length: 111540
pos_emb: 98,304
tok_emb.weight: 24,960
blocks.0.ln1.weight: 384
blocks.0.ln1.bias: 384
blocks.0.ln2.weight: 384
blocks.0.ln2.bias: 384
blocks.0.attn.in_proj_weight: 442,368
blocks.0.attn.in_proj_bias: 1,152
blocks.0.attn.out_proj.weight: 147,456
blocks.0.attn.out_proj.bias: 384
blocks.0.mlp.0.weight: 589,824
blocks.0.mlp.0.bias: 1,536
blocks.0.mlp.2.weight: 589,824
blocks.0.mlp.2.bias: 384
blocks.1.ln1.weight: 384
blocks.1.ln1.bias: 384
blocks.1.ln2.weight: 384
blocks.1.ln2.bias: 384
blocks.1.attn.in_proj_weight: 442,368
blocks.1.attn.in_proj_bias: 1,152
blocks.1.attn.out_proj.weight: 147,456
blocks.1.attn.out_proj.bias: 384
blocks.1.mlp.0.weight: 589,824
blocks.1.mlp.0.bias: 1,536
blocks.1.mlp.2.weight: 589,824
blocks.1.mlp.2.bias: 384
blocks.2.ln1.weight: 384
blocks.2.ln1.bias: 384
blocks.2.ln2.weight: 384
blocks.2.ln2.bias: 384
blocks.2.attn.in_proj_weight: 442,368
blocks.2.attn.in_proj_bias: 1,152
blocks.2.attn.out_proj.weight: 147,456
blocks.2.attn.out_proj.bias: 384
blocks.2.mlp.0.weight: 589,824
blocks.2.mlp.0.bias: 1,536
blocks.2.mlp.2.weight: 589,824
blocks.2.mlp.2.bias: 384
blocks.3.ln1.weight: 384
blocks.3.ln1.bias: 384
blocks.3.ln2.weight: 384
blocks.3.ln2.bias: 384
blocks.3.attn.in_proj_weight: 442,368
blocks.3.attn.in_proj_bias: 1,152
blocks.3.attn.out_proj.weight: 147,456
blocks.3.attn.out_proj.bias: 384
blocks.3.mlp.0.weight: 589,824
blocks.3.mlp.0.bias: 1,536
blocks.3.mlp.2.weight: 589,824
blocks.3.mlp.2.bias: 384
blocks.4.ln1.weight: 384
blocks.4.ln1.bias: 384
blocks.4.ln2.weight: 384
blocks.4.ln2.bias: 384
blocks.4.attn.in_proj_weight: 442,368
blocks.4.attn.in_proj_bias: 1,152
blocks.4.attn.out_proj.weight: 147,456
blocks.4.attn.out_proj.bias: 384
blocks.4.mlp.0.weight: 589,824
blocks.4.mlp.0.bias: 1,536
blocks.4.mlp.2.weight: 589,824
blocks.4.mlp.2.bias: 384
blocks.5.ln1.weight: 384
blocks.5.ln1.bias: 384
blocks.5.ln2.weight: 384
blocks.5.ln2.bias: 384
blocks.5.attn.in_proj_weight: 442,368
blocks.5.attn.in_proj_bias: 1,152
blocks.5.attn.out_proj.weight: 147,456
blocks.5.attn.out_proj.bias: 384
blocks.5.mlp.0.weight: 589,824
blocks.5.mlp.0.bias: 1,536
blocks.5.mlp.2.weight: 589,824
blocks.5.mlp.2.bias: 384
ln_f.weight: 384
ln_f.bias: 384
head.weight: 24,960

Total trainable parameters: 10,795,776
Step 100, Loss: 2.7967, LR: 0.000399
Step 200, Loss: 2.5411, LR: 0.000397
Step 300, Loss: 2.4117, LR: 0.000393
Step 400, Loss: 2.1451, LR: 0.000387
Step 500, Loss: 1.9782, LR: 0.000380
Step 500: Val Loss = 1.8572
New best val loss: 1.8572
Step 600, Loss: 1.8572, LR: 0.000372
Step 700, Loss: 1.7620, LR: 0.000362
Step 800, Loss: 1.6909, LR: 0.000350
Step 900, Loss: 1.6388, LR: 0.000338
Step 1000, Loss: 1.5952, LR: 0.000325
Step 1000: Val Loss = 1.5812
New best val loss: 1.5812
Step 1100, Loss: 1.5572, LR: 0.000310
Step 1200, Loss: 1.5268, LR: 0.000295
Step 1300, Loss: 1.5063, LR: 0.000278
Step 1400, Loss: 1.4795, LR: 0.000262
Step 1500, Loss: 1.4632, LR: 0.000244
Step 1500: Val Loss = 1.4954
New best val loss: 1.4954
Step 1600, Loss: 1.4445, LR: 0.000227
Step 1700, Loss: 1.4266, LR: 0.000209
Step 1800, Loss: 1.4154, LR: 0.000191
Step 1900, Loss: 1.4016, LR: 0.000173
Step 2000, Loss: 1.3905, LR: 0.000155
Step 2000: Val Loss = 1.4619
New best val loss: 1.4619
Step 2100, Loss: 1.3767, LR: 0.000138
Step 2200, Loss: 1.3691, LR: 0.000121
Step 2300, Loss: 1.3589, LR: 0.000105
Step 2400, Loss: 1.3486, LR: 0.000090
Step 2500, Loss: 1.3456, LR: 0.000075
Step 2500: Val Loss = 1.4529
New best val loss: 1.4529
Step 2600, Loss: 1.3361, LR: 0.000062
Step 2700, Loss: 1.3325, LR: 0.000049
Step 2800, Loss: 1.3276, LR: 0.000038
Step 2900, Loss: 1.3229, LR: 0.000028
Step 3000, Loss: 1.3193, LR: 0.000020
Step 3000: Val Loss = 1.4506
New best val loss: 1.4506
Step 3100, Loss: 1.3219, LR: 0.000013
Step 3200, Loss: 1.3180, LR: 0.000007
Step 3300, Loss: 1.3165, LR: 0.000003
Step 3400, Loss: 1.3159, LR: 0.000001

Generated text:
 
My dear life! My father, you speak to your lordship in me,
I'll flay you thither, yield you will not the be cause
Of all, to rule you m justice.

AUFIDIUS:
Did I live, indeed.

MONTAGUE:
Well, have you no more countrymen?

LUCIO:
Not caSt my nurse, --you
Had say say them of it mark that you?

CAPULET:
O, my lord.

LADY CAPULET:
My lord, I mean the lengthour. Why? and then?--

Nurse:
Not a deep and so. If do you go to those.

LUCENTIO:
I know not, by savage them receiving. Be it, God's joy!
Look,
```
