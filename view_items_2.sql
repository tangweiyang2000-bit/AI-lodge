select original_filename, category, subcategory, color, pattern, formality, image_path
from view_items_2
order by split_part(original_filename, '_', 2)::int;
