import { useEffect, useState, useMemo } from 'react'
import { Card, Select, Button, Row, Col, Table, Spin, message, Alert } from 'antd'
import ReactECharts from 'echarts-for-react'
import { analyticsApi, getApiErrorMessage } from '../../api'

const YEAR_OPTIONS = [2026, 2025, 2024, 2023, 2022].map((y) => ({ value: y, label: String(y) }))

/** 训练集常为跨年片段：默认选「有整年覆盖」的年份更易出图（若 train 截止较早可再切到 2024） */

/** 避免 title 与 legend 默认都挤在顶部重叠 */
const CHART_GRID = { left: '12%', right: '6%', top: 108, bottom: 56, containLabel: true }

export default function HolidayAnalysis() {
  const [year, setYear] = useState(2023)
  const [loading, setLoading] = useState(true)
  const [loadingAccuracy, setLoadingAccuracy] = useState(false)
  const [locationLabel, setLocationLabel] = useState('')
  const [bars, setBars] = useState([])
  const [accLabels, setAccLabels] = useState([])
  const [accMain, setAccMain] = useState([])
  const [accBase, setAccBase] = useState([])
  const [dataHint, setDataHint] = useState('')
  const [trainRange, setTrainRange] = useState(null)

  const load = async () => {
    setLoading(true)
    setLoadingAccuracy(false)
    setBars([])
    setAccLabels([])
    setAccMain([])
    setAccBase([])
    try {
      const loc = await analyticsApi.location()
      setLocationLabel(loc.data?.label || '')
      const loadRes = await analyticsApi.holidayLoad(year)
      setBars(loadRes.data.bars || [])
      if (loadRes.data.train_range) setTrainRange(loadRes.data.train_range)
      const loadHint = loadRes.data.hint || ''
      setDataHint(loadHint)
    } catch (e) {
      message.error(getApiErrorMessage(e))
      setBars([])
      setDataHint('')
      setTrainRange(null)
      return
    } finally {
      setLoading(false)
    }

    setLoadingAccuracy(true)
    try {
      const accRes = await analyticsApi.holidayAccuracy(year)
      setAccLabels(accRes.data.labels || [])
      setAccMain(accRes.data.accuracy_main || [])
      setAccBase(accRes.data.accuracy_baseline || [])
      const accHint = accRes.data.hint || ''
      setDataHint((prev) => {
        const parts = [prev, accHint].filter(Boolean)
        return parts.length ? [...new Set(parts)].join(' ') : ''
      })
    } catch (e) {
      message.error(`节假日准确率：${getApiErrorMessage(e)}`)
    } finally {
      setLoadingAccuracy(false)
    }
  }

  useEffect(() => {
    void load()
  }, [year])

  const chartOption = useMemo(
    () => ({
      backgroundColor: 'transparent',
      title: {
        text: `节假日期间最大负荷（训练集真值 · ${locationLabel || '—'}）`,
        left: 'center',
        top: 8,
        textStyle: { color: '#e8f4ff', fontSize: 14 },
      },
      grid: { ...CHART_GRID },
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(10,22,40,0.9)' },
      xAxis: {
        type: 'category',
        data: bars.map((b) => b.name),
        axisLabel: { color: '#e8f4ff', rotate: 28, interval: 0 },
      },
      yAxis: { type: 'value', name: 'MW', axisLabel: { color: '#e8f4ff' }, nameTextStyle: { color: '#e8f4ff' } },
      series: [
        {
          type: 'bar',
          data: bars.map((b) => b.max_load),
          itemStyle: { color: 'rgba(64,169,255,0.85)' },
          barMaxWidth: 36,
        },
      ],
    }),
    [bars, locationLabel],
  )

  const accOption = useMemo(
    () => ({
      backgroundColor: 'transparent',
      title: {
        text: '节假日首日 96 点回测平均准确率（%）',
        left: 'center',
        top: 8,
        textStyle: { color: '#e8f4ff', fontSize: 14 },
      },
      legend: {
        data: ['TCN-BiLSTM-Attention', '持久化基线'],
        top: 44,
        left: 'center',
        itemGap: 24,
        textStyle: { color: '#e8f4ff' },
      },
      grid: { ...CHART_GRID },
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(10,22,40,0.9)' },
      xAxis: {
        type: 'category',
        data: accLabels,
        axisLabel: { color: '#e8f4ff', rotate: 28, interval: 0 },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        name: '准确率 (%)',
        axisLabel: { color: '#e8f4ff' },
        nameTextStyle: { color: '#e8f4ff' },
      },
      series: [
        {
          type: 'bar',
          data: accMain,
          name: 'TCN-BiLSTM-Attention',
          itemStyle: { color: '#40a9ff' },
          barMaxWidth: 22,
        },
        {
          type: 'bar',
          data: accBase,
          name: '持久化基线',
          itemStyle: { color: '#ff4d4f' },
          barMaxWidth: 22,
        },
      ],
    }),
    [accLabels, accMain, accBase],
  )

  return (
    <div>
      <Card className="glass-card" style={{ marginBottom: 24 }}>
        <Row gutter={16}>
          <Col>
            <span style={{ color: '#e8f4ff' }}>区域：</span>
            <span style={{ color: '#91d5ff' }}>{locationLabel || '—'}</span>
          </Col>
          <Col>
            <span style={{ color: '#e8f4ff' }}>年份：</span>
            <Select style={{ width: 100 }} value={year} onChange={setYear} options={YEAR_OPTIONS} />
          </Col>
          <Col>
            <Button type="primary" className="tech-btn" onClick={load}>
              刷新
            </Button>
          </Col>
        </Row>
        <div style={{ color: 'rgba(255,255,255,0.55)', fontSize: 12, marginTop: 12 }}>
          节假日区间由「法定节假日」日历生成（访问本页时会尝试自动补全空库）；柱状图为与训练集日期有交集的节假日；准确率图为各节假日首日历史回测。
          {trainRange && (
            <span style={{ display: 'block', marginTop: 6 }}>
              当前训练集日期范围：
              <span style={{ color: '#91d5ff' }}>{trainRange.start}～{trainRange.end}</span>
              （请先选该范围内的年份，或扩充 train_data.csv）
            </span>
          )}
        </div>
      </Card>
      <Spin spinning={loading}>
        {dataHint && (
          <Alert type="warning" showIcon style={{ marginBottom: 16 }} message={dataHint} />
        )}
        <Row gutter={24}>
          <Col span={12}>
            <Card className="glass-card">
              <ReactECharts option={chartOption} style={{ height: 360 }} notMerge />
            </Card>
          </Col>
          <Col span={12}>
            <Card className="glass-card">
              <Spin spinning={loadingAccuracy}>
                <ReactECharts option={accOption} style={{ height: 360 }} notMerge />
              </Spin>
            </Card>
          </Col>
        </Row>
        <Row style={{ marginTop: 24 }}>
          <Col span={24}>
            <Card className="glass-card" title="节假日负荷极值（训练集）">
              <Table
                dataSource={bars.map((b, i) => ({ ...b, key: i }))}
                columns={[
                  { title: '节假日', dataIndex: 'name' },
                  { title: '开始', dataIndex: 'start_date' },
                  { title: '结束', dataIndex: 'end_date' },
                  { title: '最大负荷(MW)', dataIndex: 'max_load' },
                  { title: '最小负荷(MW)', dataIndex: 'min_load' },
                ]}
                size="small"
                locale={{
                  emptyText:
                    dataHint ||
                    '无数据：该年节假日与训练集无交集，或训练集路径不正确。可尝试切换年份（建议先看训练集覆盖的年份）或检查 数据/训练集/train_data.csv。',
                }}
              />
            </Card>
          </Col>
        </Row>
      </Spin>
    </div>
  )
}
