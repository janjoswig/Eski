Notes
=====

Alternative box code
--------------------

On `System.__init__()`:

```python
if box is None:
    # TODO: Check for invalid box
    self._box = np.zeros((3, 3), dtype=P_AVALUE)
    self._boxinv = np.array(self._box, copy=True, dtype=P_AVALUE   else:
    self._box = np.array(
        box, copy=True, dtype=P_AVALUE
        )
    self._boxinv = np.linalg.inv(self._box)
```