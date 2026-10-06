select original_filename, category, subcategory, color, colors, pattern, formality, image_path
from view_items_2
order by split_part(original_filename, '_', 2)::int;
