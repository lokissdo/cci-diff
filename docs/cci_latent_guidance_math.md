# CCI Latent Guidance Math

## Short Answer

Yes: right now the real CCI edit path uses `cci_latent_guidance_hook`, not
`cci_guidance_hook`.

There are two hook slots in `sd2_bld_backend.py`:

```python
cci_guidance_hook: CCIGuidanceHook | None = None
cci_latent_guidance_hook: CCILatentGuidanceHook | None = None
```

They happen at different places:

1. `cci_guidance_hook` can replace `noise_pred` before the scheduler step.
2. `cci_latent_guidance_hook` can replace `latents` after the scheduler step.

The current runner, `scripts/run_sd2_bld_cci.py`, only builds and passes
`cci_latent_guidance_hook`:

```python
cci_latent_guidance_hook = build_cci_latent_guidance_hook(backend, config, args)

result = backend.edit_image(
    ...
    cci_latent_guidance_hook=cci_latent_guidance_hook,
)
```

So:

```text
cci_guidance_hook        exists, but is not used by the current runner
cci_latent_guidance_hook is the active CCI path
```

## Old BLD Loop

BLD means Blended Latent Diffusion. The old idea is:

```text
edit inside the mask
restore the noised source outside the mask
```

At each timestep:

```text
z_t = current latent
m = binary latent mask
z_src = source image latent
eps_cfg = classifier-free-guidance noise prediction
```

The old BLD update is:

```text
z_sched = SchedulerStep(z_t, eps_cfg, t)
z_src_t = AddNoise(z_src, random_noise, t)

z_next = m * z_sched + (1 - m) * z_src_t
```

Meaning:

```text
inside mask:  keep edited latent
outside mask: restore source latent at the same noise level
```

In code this is:

```python
latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

noise_source_latents = self.scheduler.add_noise(
    source_latents,
    torch.randn_like(latents),
    timestep,
)

latents = blend_latents(latents, latent_mask, noise_source_latents)
```

And `blend_latents` is:

```python
return noise_source_latents.where(~latent_mask.bool(), latents)
```

For a binary mask, this is the same as:

```text
latents = m * latents + (1 - m) * noise_source_latents
```

## CCI Adds One Extra Step

CCI does not replace BLD. It inserts one extra operation before BLD blending:

```text
old BLD:
  scheduler step -> BLD blend

CCI-BLD:
  scheduler step -> CCI latent update -> BLD blend
```

In code:

```python
latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

if cci_latent_guidance_hook is not None:
    latents = apply_cci_latent_guidance_hook(
        latents,
        step,
        cci_latent_guidance_hook,
    )

noise_source_latents = self.scheduler.add_noise(...)
latents = blend_latents(latents, latent_mask, noise_source_latents)
```

So CCI is similar to BLD because it also uses the mask. But it does a different
job:

```text
BLD mask:
  decides where to keep edit vs restore source

CCI mask:
  decides where the concept-gradient is allowed to modify latents
```

## What Is The Latent?

Stable Diffusion does not denoise pixels directly. It denoises a compressed
latent tensor.

For a 512 x 512 image, the latent is usually:

```text
z shape = [batch, 4, 64, 64]
```

The VAE decoder maps the latent to an image:

```text
x = D(z)
```

In code:

```python
decoded = backend.vae.decode(latents / 0.18215).sample
image = decoded / 2 + 0.5
```

So:

```text
z = current latent
D(z) = decoded image-like tensor in [0, 1]
```

## Current CCI Loss

The current prototype hook is `latent_color`. For blond hair, it uses a target
RGB prior:

```text
c = [0.95, 0.78, 0.38]
```

Let:

```text
x = D(z)
M = image-space hair mask
c = target blond RGB
```

The target term uses the average color inside the mask:

```text
mean_rgb = sum(M * x) / sum(M)
L_target = ||mean_rgb - c||^2 / 3
```

This is implemented as:

```python
def _masked_mean_rgb_mse(value, target_rgb, mask):
    denominator = mask.sum().clamp_min(1.0)
    mean_rgb = (value * mask).sum(dim=(0, 2, 3)) / denominator
    return ((mean_rgb - target_rgb.view(3)) ** 2).mean()
```

Why average color instead of per-pixel color?

Per-pixel loss:

```text
force every hair pixel to become the same blond color
```

Average-color loss:

```text
make the hair region blond overall, but allow texture and variation
```

That is why `cci_hook_avg_norm2_mps` looked better than the flat painted runs.

## Outside Preservation Loss

The hook also compares the decoded current image to the decoded source image
outside the mask:

```text
x_src = D(z_src)
L_outside = sum((1 - M) * (x - x_src)^2) / (3 * sum(1 - M))
```

Implemented as:

```python
def _masked_mse(value, target, mask):
    channels = value.shape[1]
    denominator = (mask.sum() * channels).clamp_min(1.0)
    return (((value - target) ** 2) * mask).sum() / denominator
```

This says:

```text
outside the hair mask, stay close to the source image
```

## Weighted CCI Loss

The code returns these terms:

```python
GuidanceTerms(
    target=target_loss,
    preservation=outside_delta,
    leakage=outside_delta,
    classifier=target_loss,
    outside_mask=outside_delta,
)
```

Then `compose_guidance_loss` computes:

```text
L_CCI =
    w_target       * L_target
  + w_preservation * L_outside
  + w_leakage      * L_outside
  + w_classifier   * L_target
  + w_outside_mask * L_outside
```

Important limitation:

```text
right now "classifier" is still approximated by the color target
```

So this is a CCI hook prototype. A later version can replace the color target
with a real blond-hair classifier score.

## Gradient Step

Now we have:

```text
z = current latent after scheduler step
D(z) = decoded image
L_CCI = weighted CCI loss
m = latent-space mask
```

We compute the gradient:

```text
g = dL_CCI / dz
```

This means:

```text
which direction in latent space increases the bad loss?
```

To reduce the loss, move in the opposite direction:

```text
z_new = z - eta * g
```

where:

```text
eta = cci_step_size
```

## Masked Gradient

We only want the CCI update inside the mask:

```text
g_m = m * g
```

So:

```text
inside mask:  gradient is kept
outside mask: gradient becomes zero
```

Then:

```text
z_cci = z - eta * g_m
```

In code:

```python
gradient = torch.autograd.grad(loss, guided_latents)[0]

if latent_mask is not None:
    gradient = gradient * latent_mask.to(
        device=gradient.device,
        dtype=gradient.dtype,
    )

return (latents - step_size * gradient).detach()
```

## Normalized Gradient

Raw gradients can be tiny or huge depending on the VAE, timestep, and loss scale.
So we added optional normalization:

```text
g_m_norm = g_m / ||g_m||
z_cci = z - eta * g_m_norm
```

In code:

```python
if normalize_gradient:
    gradient = _normalize_gradient(gradient, gradient_eps)
```

This makes `cci_step_size` easier to tune:

```text
without normalization:
  eta depends heavily on gradient magnitude

with normalization:
  eta means "how far to move in the CCI direction"
```

That is why `--cci_normalize_grad --cci_step_size 2.0` worked much better than
small raw steps.

## Why This Still Preserves BLD

The final update in a CCI step is:

```text
1. z_sched = SchedulerStep(z_t, eps_cfg, t)
2. z_cci = z_sched - eta * normalize(m * dL_CCI/dz)
3. z_src_t = AddNoise(z_src, random_noise, t)
4. z_next = m * z_cci + (1 - m) * z_src_t
```

Step 4 is still BLD.

So the final latent outside the mask is still restored:

```text
outside mask, m = 0:

z_next = 0 * z_cci + 1 * z_src_t
       = z_src_t
```

Inside the mask:

```text
inside mask, m = 1:

z_next = 1 * z_cci + 0 * z_src_t
       = z_cci
```

That is the key distinction:

```text
BLD chooses where the edit can survive.
CCI changes what the edit is trying to become.
```

## Why It Looks Similar To BLD

It looks similar because both are masked latent operations.

But BLD and CCI answer different questions:

```text
BLD:
  Where is editing allowed?

CCI:
  In that allowed region, what concept direction should the latent move toward?
```

With no CCI hook:

```text
prompt says "add blond hair"
BLD keeps edits only in hair mask
```

With CCI hook:

```text
prompt says "add blond hair"
CCI gradient pushes decoded masked hair toward blond
BLD still keeps edits only in hair mask
```

## Two Hook Types In This Codebase

### `cci_guidance_hook`

This hook touches `noise_pred`:

```text
eps_cfg -> hook -> scheduler step
```

It could be used later for classifier guidance in noise-prediction space.
Currently the runner does not pass anything into this hook.

### `cci_latent_guidance_hook`

This hook touches `latents`:

```text
scheduler step -> hook -> BLD blend
```

This is the active hook today. It decodes the latent, computes CCI loss, takes a
gradient step, then returns updated latents.

## Current Practical Weakness

The math is working, but the mask is hard binary.

That means:

```text
the CCI gradient follows the exact CelebAMask hair shape
the BLD blend also follows the exact hard mask edge
```

So the output can look like the mask was painted onto the image.

The soft-mask plan does not change the CCI idea. It changes the operational mask:

```text
hard binary mask remains for audit
soft generation mask is used for prettier denoising and blending
```

That should reduce the hard edge and speck artifacts while keeping the causal
audit honest.
