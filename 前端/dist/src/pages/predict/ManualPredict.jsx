import { useEffect, useState, useMemo, useCallback } from 'react'
import { Card, DatePicker, Button, Table, Modal, InputNumber, Row, Col, message, Spin, Select } from 'antd'
import ReactECharts from 'echarts-for-react'
import dayjs from 'dayjs'
import { predictionApi, analyticsApi, getApiErrorMessage } from '../../api'
import { activeCompareKeys } from '../../predictionModels'

function toHourly24(points) {
  if (!points?.length) return []
  const slice = points.slice(0, 96)
  const keys = activeCompareKeys(slice[0])
  const out = []
  for (let h = 0; h < 24; h++) {
    const chunk = slice.slice(h * 4, h * 4 + 4)
    if (!chunk.length) break
    const m = chunk.reduce((s, p) => s + (p.main_mw || 0), 0) / chunk.length
    const b = chunk.reduce((s, p) => s + (p.baseline_mw || 0), 0) / chunk.length
    const row = { time: `${h}:00`, main: m, base: b, key: h }
    keys.forEach(({ key }) => {
      row[key] = chunk.reduce((s, p) => s + (Number(p[key]) || 0), 0) / chunk.length
    })
    out.push(row)
  }
  return out
}

export default function ManualPredict() {
  const [loading, setLoading] = useState(true)
  const [predDate, setPredDate] = useState(dayjs())
  const [locationLabel, setLocationLabel] = useState('')
  const [tableData, setTableData] = useState([])
  const [rawPred, setRawPred] = useState([])
  const [modifyModal, setModifyModal] = useState(null)
  const [modifyValue, setModifyValue] = useState(0)
  const [predSteps, setPredSteps] = useState(96)

  const load = async () => {
    setLoading(true)
    try {
      const loc = await analyticsApi.location()
      setLocationLabel(loc.data?.label || '')
      const start = predDate.startOf('day').toISOString()
      const res = await predictionApi.getTwoDays(start, predSteps)
      const pts = res.data.data || []
      setRawPred(pts)
      const hourly = toHourly24(pts).map((r, i) => ({ ...r, key: i }))
      setTableData(hourly)
    } catch (e) {
      message.error(getApiErrorMessage(e))
      setTableData([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [predDate, predSteps])

  const compareKeysOnData = useMemo(() => activeCompareKeys(rawPred[0]), [rawPred])

  const handleModify = useCallback((record) => {
    setModifyModal(record)
    setModifyValue(record.main)
  }, [])

  const chartOption = useMemo(() => {
    const series = [
      {
        type: 'line',
        data: tableData.map((d) => d.main),
        name: 'TCN-BiLSTM-Attention',
        smooth: true,
        lineStyle: { color: '#40a9ff', width: 2 },
      },
    ]
    compareKeysOnData.forEach(({ key, short, color }) => {
      series.push({
        type: 'line',
        data: tableData.map((d) => d[key]),
        name: short,
        smooth: true,
        lineStyle: { color, width: 1.5 },
        symbol: 'none',
      })
    })
    series.push({
      type: 'line',
      data: tableData.map((d) => d.base),
      name: '持久化模型',
      smooth: true,
      lineStyle: { color: '#ff4d4f', width: 1.5, type: 'dashed' },
      symbol: 'none',
    })
    return {
      backgroundColor: 'transparent',
      title: {
        text: `负荷预测（${predSteps} 点 · 按小时聚合 · ${locationLabel || ''}）`,
        textStyle: { color: '#e8f4ff' },
      },
      xAxis: { type: 'category', data: tableData.map((d) => d.time), axisLabel: { color: '#e8f4ff' } },
      yAxis: {
        type: 'value',
        name: 'MW',
        axisLabel: { color: '#e8f4ff' },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
      },
      series,
      legend: { type: 'scroll', textStyle: { color: '#e8f4ff' } },
    }
  }, [tableData, predSteps, locationLabel, compareKeysOnData])

  const tableColumns = useMemo(() => {
    const cols = [
      { title: '时段', dataIndex: 'time' },
      {
        title: '主模型(MW)',
        dataIndex: 'main',
        render: (v) => (v != null ? Number(v).toFixed(2) : '—'),
      },
    ]
    compareKeysOnData.forEach(({ key, short }) => {
      cols.push({
        title: `${short}(MW)`,
        dataIndex: key,
        render: (v) => (v != null ? Number(v).toFixed(2) : '—'),
      })
    })
    cols.push({
      title: '持久化(MW)',
      dataIndex: 'base',
      render: (v) => (v != null ? Number(v).toFixed(2) : '—'),
    })
    cols.push({
      title: '操作',
      render: (_, r) => (
        <Button type="link" size="small" onClick={() => handleModify(r)}>
          修正主模型展示值
        </Button>
      ),
    })
    return cols
  }, [compareKeysOnData, handleModify])

  const confirmModify = () => {
    if (!modifyModal) return
    const oldVal = modifyModal.main
    Modal.confirm({
      title: '确认修改',
      content: `是否将 ${modifyModal.time} 的主模型均值由 ${oldVal.toFixed(2)} MW 改为 ${modifyValue} MW？（仅本地展示，未写入服务端）`,
      okText: '确定',
      cancelText: '取消',
      onOk: () => {
        setTableData((prev) =>
          prev.map((r) => (r.key === modifyModal.key ? { ...r, main: modifyValue } : r)),
        )
        setModifyModal(null)
        message.success('已更新本地展示值')
      },
    })
  }

  return (
    <div>
      <Card className="glass-card" style={{ marginBottom: 24 }}>
        <Row gutter={16} align="middle">
          <Col>
            <span style={{ color: '#e8f4ff' }}>区域：</span>
            <span style={{ color: '#91d5ff' }}>{locationLabel || '—'}</span>
          </Col>
          <Col>
            <span style={{ color: '#e8f4ff' }}>预测日期：</span>
            <DatePicker value={predDate} onChange={(d) => setPredDate(d || dayjs())} />
          </Col>
          <Col>
            <span style={{ color: '#e8f4ff' }}>长度：</span>
            <Select
              value={predSteps}
              onChange={setPredSteps}
              style={{ width: 200 }}
              options={[
                { value: 96, label: '快速 96 点' },
                { value: 192, label: '完整 192 点' },
              ]}
            />
          </Col>
          <Col>
            <Button type="primary" className="tech-btn" onClick={load}>
              刷新预测
            </Button>
          </Col>
        </Row>
      </Card>
      <Spin spinning={loading}>
        <Card className="glass-card" style={{ marginBottom: 24 }}>
          <ReactECharts option={chartOption} style={{ height: 350 }} notMerge />
        </Card>
        <Card className="glass-card" title="预测负荷特性（首日 24 小时均值）">
          <Table
            dataSource={tableData}
            columns={tableColumns}
            pagination={{ pageSize: 12 }}
            size="small"
            scroll={{ x: 400 + compareKeysOnData.length * 120 }}
            locale={{ emptyText: rawPred.length ? '' : '无预测数据' }}
          />
        </Card>
      </Spin>

      <Modal
        title="修正负荷值（本地）"
        open={!!modifyModal}
        onOk={confirmModify}
        onCancel={() => setModifyModal(null)}
        okText="确定"
      >
        {modifyModal && (
          <div>
            <p>时段：{modifyModal.time}</p>
            <p>当前主模型均值：{modifyModal.main.toFixed(2)} MW</p>
            <p>修改为：</p>
            <InputNumber value={modifyValue} onChange={setModifyValue} min={0} step={0.1} style={{ width: 120 }} />
          </div>
        )}
      </Modal>
    </div>
  )
}
