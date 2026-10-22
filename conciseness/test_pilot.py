import unittest
import torch
from pilot import Spline, fit_model


class GeometryTests(unittest.TestCase):
    def test_upstream_spline_interpolates_and_keeps_straight_line_straight(self):
        u = torch.arange(5, dtype=torch.float64)[:, None]
        y = torch.cat([2*u+1, -3*u+4], 1)
        curve = Spline(u, y, bc='natural')
        query = torch.linspace(0,4,17,dtype=torch.float64)[:,None]
        self.assertTrue(torch.allclose(curve.evaluate(query),torch.cat([2*query+1,-3*query+4],1),atol=1e-9))

    def test_question_offsets_do_not_define_geometry(self):
        generator=torch.Generator().manual_seed(7)
        x=torch.randn(12,5,20,generator=generator,dtype=torch.float64)
        offsets=100*torch.randn(12,1,20,generator=generator,dtype=torch.float64)
        first,_=fit_model(x,8)
        second,_=fit_model(x+offsets,8)
        self.assertTrue(torch.allclose(first['basis']@first['basis'].T,
                                       second['basis']@second['basis'].T,atol=1e-8))
        self.assertTrue(torch.allclose(first['centroids']@first['basis'].T,
                                       second['centroids']@second['basis'].T,atol=1e-8))

    def test_affine_data_does_not_create_spurious_dimensions(self):
        u=torch.arange(5,dtype=torch.float64)
        v=torch.arange(1,9,dtype=torch.float64)
        x=(u[:,None]*v)[None].repeat(10,1,1)
        fit,_=fit_model(x,64)
        self.assertEqual(fit['basis'].shape[1],1)


if __name__=='__main__': unittest.main()
