# Challenge Overview

You are provided with a disk image of someone who was unfortunate enough to delete a pin that was important to them. Your task is to retrieve the pin. 

## Objective

- Recover deleted or hidden files from the disk image or directly locate the pin.
- Use forensic tools to carve data.
- Find and decode the flag.

## How to get there

Run ``photorec`` or ``strings`` to find the file or the content of the deleted file in the where_is_my_pin.img.

This will yield either the file containing the string or the string that looks very much like something that is base64 encoded. 

``SV80bV80Y3R1YTExeV93aDR0X2M0bl9iZV91c2VkXzRzX3RoM19wMU5fZjByX3RoM19yMGIwdAo=``

If you now toss this into a base64 decoder such as the one your linux machine ships with: 

`echo SV80bV80Y3R1YTExeV93aDR0X2M0bl9iZV91c2VkXzRzX3RoM19wMU5fZjByX3RoM19yMGIwdAo=" | base64 -d` 

You get the flag content: **I_4m_4ctua11y_wh4t_c4n_be_used_4s_th3_p1N_f0r_th3_r0b0t**
