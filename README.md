A small demo to show the performances of [Performers](https://arxiv.org/abs/2009.14794) and [Transformers](https://arxiv.org/abs/1706.03762) on simple cases

```python
import torch
```
Optimal Case: Large number of time points, less number of hidden dimension


```python
x = torch.rand(32,2048,64)
print(x.shape)
```

torch.Size([32, 2048, 64])
    


```python
q = x
q.shape
```




torch.Size([32, 2048, 64])




```python
k = x.permute(0,2,1)
k.shape
```




torch.Size([32, 64, 2048])




```python
v = x
v.shape
```




torch.Size([32, 2048, 64])




```python
import time
tic = time.time()
attn = torch.matmul(q,k)
res = torch.matmul(attn,v)
toc = time.time()
print(res.shape)
transformer_time = toc-tic
print(transformer_time)
```

torch.Size([32, 2048, 64])


0.5003597736358643
    


```python
import time
tic = time.time()
mat = torch.matmul(k,v)
res = torch.matmul(q,mat)
toc = time.time()
print(res.shape)
performer_time = toc-tic
print(performer_time)
```

torch.Size([32, 2048, 64])


0.03099536895751953
    


```python
print('improvement:',str(round(transformer_time/performer_time,2)),'times faster')
```

improvement: 16.14 times faster
    

Worst Case: Less number of time points, large number of hidden dimension


```python
x = torch.rand(32,64,2048)
print(x.shape)
```

torch.Size([32, 64, 2048])
    


```python
q = x
q.shape
```



torch.Size([32, 64, 2048])




```python
k = x.permute(0,2,1)
k.shape
```




torch.Size([32, 2048, 64])




```python
v = x
v.shape
```




torch.Size([32, 64, 2048])




```python
import time
tic = time.time()
attn = torch.matmul(q,k)
res = torch.matmul(attn,v)
toc = time.time()
print(res.shape)
transformer_time = toc-tic
print(transformer_time)
```

torch.Size([32, 64, 2048])


0.0319976806640625
    


```python
import time
tic = time.time()
mat = torch.matmul(k,v)
res = torch.matmul(q,mat)
toc = time.time()
print(res.shape)
performer_time = toc-tic
print(performer_time)
```

torch.Size([32, 64, 2048])


0.5034613609313965
    


```python
print('deterioration:',str(round(performer_time/transformer_time,2)),'times slower')
```

deterioration: 15.73 times slower
    

Balanced Case: Equal number of time points, hidden dimensions


```python
x = torch.rand(32,256,256)
print(x.shape)
```

torch.Size([32, 256, 256])
    


```python
q = x
q.shape
```




torch.Size([32, 256, 256])




```python
k = x.permute(0,2,1)
k.shape
```




torch.Size([32, 256, 256])




```python
v = x
v.shape
```




torch.Size([32, 256, 256])




```python
import time
tic = time.time()
attn = torch.matmul(q,k)
res = torch.matmul(attn,v)
toc = time.time()
print(res.shape)
transformer_time = toc-tic
print(transformer_time)
```

torch.Size([32, 256, 256])


0.030994653701782227
    


```python
import time
tic = time.time()
mat = torch.matmul(k,v)
res = torch.matmul(q,mat)
toc = time.time()
print(res.shape)
performer_time = toc-tic
print(performer_time)
```

torch.Size([32, 256, 256])


0.03600144386291504
    
