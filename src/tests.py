import unittest

from .lsi import LSI


class LSITest(unittest.TestCase):
    def test(self):
        documents = [
            'Shipment of gold damaged in a fire.',
            'Delivery of silver arrived in a silver truck.',
            'Shipment of gold arrived in a truck.'
        ]
        q = 'gold silver truck'
        lsi = LSI(documents, q)
        ranking = lsi.process()

        self.assertTrue(all(ranking == [2, 3, 1]))


if __name__ == '__main__':
    unittest.main()
