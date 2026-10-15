select original_filename, category, subcategory, sleeve, color, shade, colors, pattern, formality, image_path,
       -- first 3 of the 512 CLIP vector values, rounded to 3 decimals
       array[round((clip_vec::real[])[1]::numeric, 3),
             round((clip_vec::real[])[2]::numeric, 3),
             round((clip_vec::real[])[3]::numeric, 3)] as vec_preview
from view_items_2
order by split_part(original_filename, '_', 2)::int;
