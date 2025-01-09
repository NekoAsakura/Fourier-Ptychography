import torch


def get_args(
    m, n, aberr, array_size, wave_len, sp_size, p_size, NA, led_gap, led_height
):
    device = torch.device("mps")
    scale = int(sp_size / p_size)
    m1 = m // scale
    n1 = n // scale
    k0 = 2 * torch.pi / wave_len

    xloc, yloc = torch.meshgrid(
        torch.arange(array_size, device=device),
        torch.arange(array_size, device=device),
        indexing="ij",
    )
    xloc = ((xloc - (array_size - 1) / 2) * led_gap).reshape(-1)
    yloc = ((yloc - (array_size - 1) / 2) * led_gap).reshape(-1)

    kx_rel = -torch.sin(torch.atan(xloc / led_height))
    ky_rel = -torch.sin(torch.atan(yloc / led_height))
    kx = k0 * kx_rel
    ky = k0 * ky_rel

    dkx = 2 * torch.pi / (n * p_size)
    dky = 2 * torch.pi / (m * p_size)
    cutoff = NA * k0
    kmax = torch.pi / sp_size

    kxm, kym = torch.meshgrid(
        torch.linspace(-kmax, kmax, n1, device=device),
        torch.linspace(-kmax, kmax, m1, device=device),
        indexing="ij",
    )

    ctf = (kxm**2 + kym**2) < (cutoff**2)

    pupil = None
    if aberr is not None:
        z = aberr
        kzm_sq = k0**2 - kxm**2 - kym**2
        kzm = torch.sqrt(torch.clamp(kzm_sq, min=0.0)) + 1j * torch.sqrt(
            torch.clamp(-kzm_sq, min=0.0)
        )
        pupil = torch.exp(1j * z * torch.real(kzm)) * torch.exp(
            -abs(z) * torch.abs(torch.imag(kzm))
        )

    return ctf.float(), pupil, kx, ky, dkx, dky, m1, n1


def gseq(array_size):
    assert array_size % 2 == 1, "NotImplemented: only support odd array_size yet."
    n = (array_size + 1) // 2
    seq = torch.zeros((2, array_size**2), dtype=torch.int32)
    seq[0, 0], seq[1, 0] = n, n
    dx, dy = +1, -1
    stepx, stepy = +1, -1
    direction = +1
    counter = 0

    for i in range(1, array_size**2):
        counter += 1
        if direction == +1:
            seq[0, i] = seq[0, i - 1] + dx
            seq[1, i] = seq[1, i - 1]
            if counter == abs(stepx):
                counter = 0
                direction *= -1
                dx *= -1
                stepx *= -1
                if stepx > 0:
                    stepx += 1
                else:
                    stepx -= 1
        else:
            seq[0, i] = seq[0, i - 1]
            seq[1, i] = seq[1, i - 1] + dy
            if counter == abs(stepy):
                counter = 0
                direction *= -1
                dy *= -1
                stepy *= -1
                if stepy > 0:
                    stepy += 1
                else:
                    stepy -= 1

    seq = (seq[0, :] - 1) * array_size + seq[1, :] - 1
    return seq
