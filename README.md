# Latent Semantic Indexing

Latent semantic indexing (LSI) is an indexing and retrieval method that uses a
mathematical technique called singular value decomposition (SVD) to identify
patterns in the relationships between the terms and concepts contained in an
unstructured collection of text.

## Example

```
documents = [
    'Shipment of gold damaged in a fire.',
    'Delivery of silver arrived in a silver truck.',
    'Shipment of gold arrived in a truck.'
]
q = 'gold silver truck'

lsi = LSI(documents, q)
ranking = lsi.process()

print(ranking)
```

Result:

```
[2, 1, 3]
```

## References

1. [SVD and LSI Tutorial 4: Latent Semantic Indexing (LSI) How-to Calculations](http://manuel.midoriparadise.com/public_html/svd-lsi-tutorial.pdf)
