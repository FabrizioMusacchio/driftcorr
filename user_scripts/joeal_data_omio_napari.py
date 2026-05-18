# %% IMPORTS
import omio as om
import pprint
om.hello_world()
# %% LOAD DATA
#fname = "/Users/husker/Workspace/Joel Imaris Reg Corrections/TIF files/registered_stack_5D_251024_P4_750um_pup1_test2_v2.2.tif"
fname = "/Users/husker/Workspace/Joel Imaris Reg Corrections/TIF files/registered_stack_5D_251023_P4_470um_pup1_v1.tif"

image_lazy, metadata_lazy = om.imread(fname, zarr_store="disk")
print(f"Lazy image shape: {image_lazy.shape}")
# %% NAPARI
#metadata_lazy["PhysicalSizeZ"] = 5  # in microns

om.open_in_napari(image_lazy, metadata_lazy, fname, zarr_mode="zarr_nodask")
#om.open_in_napari(image_lazy, metadata_lazy, fname)
# %% CLEANUP
om.cleanup_omio_cache(fname, full_cleanup=True)
# %% END
