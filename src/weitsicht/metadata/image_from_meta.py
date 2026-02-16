# -----------------------------------------------------------------------
# Copyright 2026 Martin Wieser
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------

"""Build images from metadata (IOR + EOR) with optional caching/tracing."""

from __future__ import annotations

from pyproj import CRS

from weitsicht.image.perspective import ImagePerspective
from weitsicht.metadata.camera_estimator_metadata import ior_from_meta
from weitsicht.metadata.eor_from_meta import eor_from_meta
from weitsicht.metadata.metadata_results import (
    EORFromMetaResult,
    ImageFromMetaResult,
    ImageFromMetaResultSuccess,
    IORFromMetaResult,
    MetadataIssue,
)
from weitsicht.metadata.tag_systems.tag_base import MetaTagAll, MetaTagsBase
from weitsicht.utils import ResultFailure


class ImageFromMetaBuilder:
    """Build an :class:`~weitsicht.image.perspective.ImagePerspective` from metadata.

    The builder caches intermediate results (IOR/EOR) and provides a single place to add
    tracing/debug information in the future.
    """

    def __init__(self, tags: MetaTagsBase | MetaTagAll):
        self.tags: MetaTagAll = tags.get_all() if isinstance(tags, MetaTagsBase) else tags
        self._ior: IORFromMetaResult | None = None
        self._eor: EORFromMetaResult | None = None

    def ior(self) -> IORFromMetaResult:
        if self._ior is None:
            self._ior = ior_from_meta(tags_ior=self.tags.ior_base, tags_ior_extended=self.tags.ior_extended)
        return self._ior

    def eor(
        self,
        crs: CRS | None = None,
        vertical_ref: str = "ellipsoidal",
        height_rel: float = 0.0,
    ) -> EORFromMetaResult:
        if self._eor is None:
            self._eor = eor_from_meta(tags=self.tags, crs=crs, vertical_ref=vertical_ref, height_rel=height_rel)
        return self._eor

    def image(
        self,
        crs: CRS | None = None,
        vertical_ref: str = "ellipsoidal",
        height_rel: float = 0.0,
    ) -> ImageFromMetaResult:
        ior_res = self.ior()
        eor_res = self.eor(crs=crs, vertical_ref=vertical_ref, height_rel=height_rel)

        if ior_res.ok is False or eor_res.ok is False:
            errors: list[str] = []
            issues: set[MetadataIssue] = set()
            if ior_res.ok is False:
                errors.append(ior_res.error)
                issues.update(ior_res.issues)
            if eor_res.ok is False:
                errors.append(eor_res.error)
                issues.update(eor_res.issues)
            return ResultFailure(ok=False, error="; ".join(errors), issues=issues)

        image = ImagePerspective(
            width=ior_res.width,
            height=ior_res.height,
            camera=ior_res.camera,
            position=eor_res.position,
            orientation=eor_res.orientation,
            crs=eor_res.crs,
        )
        return ImageFromMetaResultSuccess(ok=True, image=image, ior=ior_res, eor=eor_res)


def image_from_meta(
    tags: MetaTagsBase | MetaTagAll,
    crs: CRS | None = None,
    vertical_ref: str = "ellipsoidal",
    height_rel: float = 0.0,
) -> ImageFromMetaResult:
    """Build an image (camera + EOR) from metadata.

    :param tags: Tag resolver instance or already-resolved tag container.
    :type tags: MetaTagsBase | MetaTagAll
    :param crs: Optional override CRS for interpreting GPS tags, defaults to ``None``.
    :type crs: CRS | None
    :param vertical_ref: Vertical reference mode passed to :func:`eor_from_meta`, defaults to ``ellipsoidal``.
    :type vertical_ref: str
    :param height_rel: Reference height (meters) for ``vertical_ref='relative'``, defaults to ``0.0``.
    :type height_rel: float
    :return: Successful image build result or a failure result.
    :rtype: ImageFromMetaResult
    """

    return ImageFromMetaBuilder(tags).image(crs=crs, vertical_ref=vertical_ref, height_rel=height_rel)
