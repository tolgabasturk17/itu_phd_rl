# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: air_traffic.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x11\x61ir_traffic.proto\x12\"edu.itu.cs.atcas.infrastructure.ml\"\x93\x01\n\x12\x41irTrafficResponse\x12\x16\n\x0esector_density\x18\x01 \x03(\x01\x12\x1a\n\x12loss_of_separation\x18\x02 \x03(\x01\x12\x17\n\x0fspeed_deviation\x18\x03 \x03(\x01\x12\x1a\n\x12\x61irflow_complexity\x18\x04 \x03(\x01\x12\x14\n\x0csector_entry\x18\x05 \x03(\x01\"-\n\x11\x41irTrafficRequest\x12\x18\n\x10\x63onfiguration_id\x18\x01 \x01(\t2\x98\x01\n\x11\x41irTrafficService\x12\x82\x01\n\x11GetAirTrafficInfo\x12\x35.edu.itu.cs.atcas.infrastructure.ml.AirTrafficRequest\x1a\x36.edu.itu.cs.atcas.infrastructure.ml.AirTrafficResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'air_traffic_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_AIRTRAFFICRESPONSE']._serialized_start=58
  _globals['_AIRTRAFFICRESPONSE']._serialized_end=205
  _globals['_AIRTRAFFICREQUEST']._serialized_start=207
  _globals['_AIRTRAFFICREQUEST']._serialized_end=252
  _globals['_AIRTRAFFICSERVICE']._serialized_start=255
  _globals['_AIRTRAFFICSERVICE']._serialized_end=407
# @@protoc_insertion_point(module_scope)
