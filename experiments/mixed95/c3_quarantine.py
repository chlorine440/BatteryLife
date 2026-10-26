"""Explicit, fingerprinted exception for ONE user-approved non-feature C3 row.

Never edits/repackages the workbook. The original CRC failure remains recorded.
Every byte outside the exact quarantined XML row is retained, and the complete
remaining XML is parsed strictly, including all rows after the damaged record.
"""
import hashlib
import io
import re
import struct
from xml.etree import ElementTree as ET
import zipfile
import zlib

from experiments.comac95.data import NS, _column_index
from experiments.mixed95.c3_evaluate import sha

FILE = 'C3-300-1.xlsx'
FILE_SHA = '466abfc37d7a418dfc8b45dd9c8ec692795e8702622a684d43cb632b628c2d5f'
PART = 'xl/worksheets/sheet6.xml'
ROW = 56394
ROW_SHA = 'da2d6b66cd0c26e4906b437d9d5ab7da545b8e86d1ed27eb6459265225915714'


def remove_exact_nonfeature_row(payload, row_number, expected_sha, expected_step=15):
    pattern = rb'<row\b[^>]*\br="'+str(row_number).encode()+rb'"[^>]*>.*?</row>'
    matches = list(re.finditer(pattern, payload, re.S))
    if len(matches) != 1:
        raise ValueError('expected exactly one quarantined row')
    match = matches[0]; row = match.group()
    if hashlib.sha256(row).hexdigest() != expected_sha:
        raise ValueError('quarantined row fingerprint changed')
    step = re.search(rb'<c r="C'+str(row_number).encode()+rb'"><v>(.*?)</v>',row)
    if step is None or int(step.group(1)) != expected_step or expected_step == 10:
        raise ValueError('only the approved NON-charging row may be quarantined')
    clean = payload[:match.start()]+payload[match.end():]
    return clean, dict(xml_row=row_number,row_sha256=expected_sha,step=expected_step,
        byte_start=match.start(),removed_bytes=len(row),bytes_outside_row_unchanged=True,
        raw_member_sha256=hashlib.sha256(payload).hexdigest(),
        derived_member_sha256=hashlib.sha256(clean).hexdigest())


def quarantined_sheet_rows(book, sheet, log):
    path = book.path
    if path.name != FILE or sha(path) != FILE_SHA or book.sheets[sheet] != PART:
        raise ValueError('quarantine requested for an unapproved file or worksheet')
    info = book.archive.getinfo(PART)
    if info.compress_type != zipfile.ZIP_DEFLATED or info.flag_bits & 1:
        raise ValueError('unexpected compression or encrypted member')
    # Read the identified raw DEFLATE payload with independent CRC computation.
    # This does NOT relabel the failed archive CRC as successful.
    with path.open('rb') as stream:
        stream.seek(info.header_offset)
        header = stream.read(30)
        if header[:4] != b'PK\x03\x04':
            raise ValueError('invalid ZIP local header')
        name_length, extra_length = struct.unpack_from('<HH',header,26)
        stream.seek(name_length+extra_length,1)
        compressed = stream.read(info.compress_size)
    payload = zlib.decompress(compressed,-15)
    actual_crc = zlib.crc32(payload)
    if len(payload) != info.file_size or actual_crc != 0x87965d7c or info.CRC != 0xbf763b08:
        raise ValueError('unexpected corruption signature; exception cannot be reused')
    clean, record = remove_exact_nonfeature_row(payload,ROW,ROW_SHA)
    del payload, compressed
    record.update(source_file=FILE,file_sha256=FILE_SHA,sheet=sheet,part=PART,
        raw_crc_pass=False,actual_crc=f'{actual_crc:08x}',recorded_crc=f'{info.CRC:08x}',
        record_id=656513,local_cycle=57,physical_cycle=257,
        user_approved_quarantine=True,raw_file_modified=False,feature_step=10)
    log.append(record)
    parser = ET.iterparse(io.BytesIO(clean),events=('start','end'))
    _,root = next(parser)
    count=after=last=charge_after=0
    for event,element in parser:
        if event != 'end' or element.tag.rsplit('}',1)[-1] != 'row':
            continue
        n=int(element.attrib['r']);count+=1;last=max(last,n);after+=int(n>ROW)
        if n == ROW:
            raise ValueError('quarantined row reappeared')
        values={}
        for cell in element:
            ref=cell.attrib.get('r','');value=cell.find('x:v',NS)
            if not ref or value is None or value.text is None:
                continue
            raw=value.text
            if cell.attrib.get('t')=='s':raw=book.strings[int(raw)]
            values[_column_index(ref)]=raw
        if n>ROW and values.get(2)=='10':charge_after+=1
        yield values
        element.clear();root.clear()
    if (count,last,after)!=(200041,200042,143648) or charge_after<=0:
        raise ValueError('rows after quarantine were truncated or lost')
    record.update(remaining_rows=count,last_original_xml_row=last,rows_after_quarantine=after,
                  feature_records_after_quarantine=charge_after,strict_remaining_xml_pass=True)
