Lazyness
====

It would be very nice to form our operators and variables such that the updates could be written simply as:

```c++
// need to copy out updates to boundary points since they may be
// overriden
copy(Px, U | take(T));
ddx = (Ox + Bx) * U;
copy(U | take(Tinv), dPx);

// similar boundary treatments and then 
ddy = ((Oy + By) * U) 
ddz = ((Ox + Bz) * U) 

rhs = ddx + (ddy | transpose(1)) + (ddz | transpose(2));
```

Something like this could be accomplished using ranges.  We'll stick with `range-v3` since `std::ranges` don't have all
the functionality we need (like `zip`).

Some implications

* The `take` (or probably `take_if` or `filter`) view might not be the most efficient way of doing
things.  Likely we will need some kind of `restrict` adaptor

* Our operators will need to take ranges, specifically `random_access_ranges`, as their arguments

* `transpose` will be another custom adaptor but should use the concept of configurable fast/slow indices rather
than hardcoding things like above