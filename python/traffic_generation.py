import json
from abc import abstractmethod, ABC
from datetime import timedelta, datetime
import xml.etree.cElementTree as ET
from xml.etree.ElementTree import Element, ElementTree

import influxdb_client
from influxdb_client.client.flux_table import FluxTable
from influxdb_client.client.write_api import SYNCHRONOUS

# Credentals removed for security reasons

org = ""
token = ""
url = ""


class TripInterval:
    def __init__(self, begin, end):
        self.trips: dict[str, TripItem] = {}
        self.begin = begin
        self.end = end

    def add_trip(self, data):
        if data.get_id() not in self.trips:
            self.trips[data.get_id()] = data
        else:
            self.trips[data.get_id()].count += data.count

    def to_xml(self, additional: Element):
        interval = ET.SubElement(additional, 'interval', {
            'begin': str(self.begin),
            'end': str(self.end)
        })
        for trip in self.trips:
            self.trips[trip].to_xml(interval)


class TrafficGen:
    def __init__(self):
        self.client = influxdb_client.InfluxDBClient(
            url=url,
            token=token,
            org=org
        ).query_api()
        self.mapping = json.load(open('trip_generation/detector_mapping.json'))

    def get_data(self):
        interval = timedelta(minutes=30)
        # Define the parameters
        params = {

            "timeRangeStart": datetime.now() - timedelta(days=5),
            "timeRangeStop": datetime.now() - timedelta(days=1),
            "windowPeriod": interval

        }

        query = '''from(bucket: "trafficdata")
          |> range(start: timeRangeStart, stop: timeRangeStop)
          |> filter(fn: (r) => r["detectorState"] == "o.k." or r["detectorState"] == "p.o.k.")
          |> filter(fn: (r) => r["_field"] == "count")
          |> filter(fn: (r) => r["relId"] == "K130_D03_D4" or
                               r["relId"] == "K130_D31_DB13lab_1" or
                               r["relId"] == "K130_D30_DB13rab_1" or
                               r["relId"] == "K130_D29_DB1lab_1" or
                               r["relId"] == "K130_D28_DB1rab_1" or
                               r["relId"] == "K130_D27_D3b" or
                               r["relId"] == "K130_D26_D3a" or
                               r["relId"] == "K130_D21_ZD3LKW" or
                               r["relId"] == "K130_D20_ZD3KFZ" or
                               r["relId"] == "K130_D19_ZD2LKW" or
                               r["relId"] == "K130_D18_ZD2KFZ" or
                               r["relId"] == "K130_D17_ZD1LKW" or
                               r["relId"] == "K130_D16_ZD1KFZ" or
                               r["relId"] == "K130_D13_D35b" or
                               r["relId"] == "K130_D09_ZD4LKW" or
                               r["relId"] == "K130_D08_ZD4KFZ" or
                               r["relId"] == "K130_D06_D37KFZ" or
                               r["relId"] == "K130_D05_D6" or
                               r["relId"] == "K130_D04_D35a" or
                               r["relId"] == "K330_D52_DB12ab_2" or
                               r["relId"] == "K330_D12_DB5ab_1" or
                               r["relId"] == "K330_D54_D6b" or
                               r["relId"] == "K330_D10_D11_alle" or
                               r["relId"] == "K330_D14_DB1rab_1" or
                               r["relId"] == "K330_D15_DB1lab_1" or
                               r["relId"] == "K330_D17_D12b" or
                               r["relId"] == "K330_D22_D3b" or
                               r["relId"] == "K330_D18_D12a" or
                               r["relId"] == "K330_D06_D6a" or
                               r["relId"] == "K330_D13_DB6ab_1" or
                               r["relId"] == "K330_D09_D4" or
                               r["relId"] == "K330_D02_D2" or
                               r["relId"] == "K330_D07_D7" or
                               r["relId"] == "K330_D53_D3" or
                               r["relId"] == "K330_D11_D11_LKW" or
                               r["relId"] == "K330_D04_D4a" or
                               r["relId"] == "K330_D08_D4b" or
                               r["relId"] == "K330_D03_D3a" or
                               r["relId"] == "K330_D05_D5")
          |> group(columns: ["relId", "_measurement"])
          |> aggregateWindow(every: windowPeriod, fn: mean, createEmpty: false)
          |> group(columns: ["_time"])'''

        result = self.client.query(org=org, query=query, params=params)

        begin = 0
        data = ET.Element('data')
        for table in result:
            end = begin + round(interval.total_seconds())
            trip_interval = self._parse_table(table, begin, interval)

            trip_interval.to_xml(data)
            begin = end

        et = ElementTree(data)
        et.write("edge_relations.xml")

    def _parse_table(self, table: FluxTable, begin, interval) -> TripInterval:

        trip_interval = TripInterval(begin, begin + round(interval.total_seconds()))
        for row in table.records:
            if row['relId'] not in self.mapping:
                continue

            found_mappings = self.mapping[row['relId']]
            for found_mapping in found_mappings:
                if found_mapping['type'] == 'edge':
                    trip_interval.add_trip(TripEdge(found_mapping['id'], row['_value'], interval))
                else:
                    trip_interval.add_trip(TripEdgeRelation(found_mapping['from'], found_mapping['to'], row['_value'], interval))

        return trip_interval


class TripItem(ABC):
    def __init__(self):
        self.count: float = None

    @abstractmethod
    def to_xml(self, interval: Element):
        pass

    @abstractmethod
    def get_id(self):
        pass


class TripEdge(TripItem):
    def get_id(self):
        return self.edge_id

    def to_xml(self, interval: Element):
        ratio = timedelta(hours=1) / self.interval
        ET.SubElement(interval, 'edge', {
            'id': self.edge_id,
            'count': str(self.count / ratio)
        })
        pass

    def __init__(self, edge_id, count, interval: timedelta):
        super().__init__()
        self.edge_id = edge_id
        self.count = count
        self.interval = interval


class TripEdgeRelation(TripItem):
    def get_id(self):
        return self.from_edge + self.to_edge

    def to_xml(self, interval: Element):
        ratio = timedelta(hours=1) / self.interval
        ET.SubElement(interval, 'edgeRelation', {
            'from': self.from_edge,
            'to': self.to_edge,
            'count': str(self.count / ratio)
        })

    def __init__(self, from_edge, to_edge, count, interval: timedelta):
        super().__init__()
        self.from_edge = from_edge
        self.to_edge = to_edge
        self.count = count
        self.interval = interval
