import { useState, useEffect } from 'react'
import { Card, Table, Button, Modal, Form, Input, Select, DatePicker, message } from 'antd'
import { holidayApi } from '../../api'
import dayjs from 'dayjs'

export default function HolidayConfig() {
  const [data, setData] = useState([])
  const [modalOpen, setModalOpen] = useState(false)
  const [editingId, setEditingId] = useState(null)
  const [form] = Form.useForm()

  const loadData = async () => {
    try {
      const res = await holidayApi.list()
      const list = (res.data?.data || []).map((d, i) => ({ ...d, key: d.id || i }))
      setData(list)
    } catch {
      setData([{ key: 1, id: '1', name: '春节', start_date: '2024-02-10', end_date: '2024-02-17', type: '法定' }])
    }
  }

  useEffect(() => { loadData() }, [])

  const handleAdd = () => {
    setEditingId(null)
    form.resetFields()
    setModalOpen(true)
  }

  const handleEdit = (record) => {
    setEditingId(record.id || record.key)
    const start = record.start_date || record.start
    const end = record.end_date || record.end
    form.setFieldsValue({
      name: record.name,
      start_date: start ? dayjs(start) : undefined,
      end_date: end ? dayjs(end) : undefined,
      type: record.type || '法定',
    })
    setModalOpen(true)
  }

  const handleDelete = (record) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定删除节假日「${record.name}」吗？`,
      okText: '确定',
      cancelText: '取消',
      onOk: () => {
        setData(prev => prev.filter(d => (d.id || d.key) !== (record.id || record.key)))
        message.success('已删除')
      },
    })
  }

  const handleSubmit = async () => {
    try {
      const v = await form.validateFields()
      const startStr = dayjs(v.start_date).format('YYYY-MM-DD')
      const endStr = dayjs(v.end_date).format('YYYY-MM-DD')
      if (editingId) {
        setData(prev => prev.map(d => (d.id || d.key) === editingId ? { ...d, name: v.name, start_date: startStr, end_date: endStr, type: v.type } : d))
        message.success('已更新')
      } else {
        try {
          await holidayApi.create({ name: v.name, start_date: startStr, end_date: endStr, type: v.type })
        } catch {}
        setData(prev => [...prev, { key: Date.now(), name: v.name, start_date: startStr, end_date: endStr, type: v.type }])
        message.success('已添加')
      }
      setModalOpen(false)
    } catch (e) {
      if (e.errorFields) return
    }
  }

  return (
    <Card className="glass-card" title="节假日信息管理">
      <Button type="primary" className="tech-btn" style={{ marginBottom: 16 }} onClick={handleAdd}>新增节假日</Button>
      <Table
        dataSource={data}
        columns={[
          { title: '名称', dataIndex: 'name' },
          { title: '开始日期', dataIndex: 'start_date', render: (_, r) => r.start_date || r.start },
          { title: '结束日期', dataIndex: 'end_date', render: (_, r) => r.end_date || r.end },
          { title: '类型', dataIndex: 'type' },
          { title: '操作', render: (_, r) => (
            <>
              <Button type="link" size="small" onClick={() => handleEdit(r)}>编辑</Button>
              <Button type="link" size="small" danger onClick={() => handleDelete(r)}>删除</Button>
            </>
          ) },
        ]}
      />
      <Modal title={editingId ? '编辑节假日' : '新增节假日'} open={modalOpen} onOk={handleSubmit} onCancel={() => setModalOpen(false)} okText="确定">
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="名称" rules={[{ required: true }]}>
            <Input placeholder="如：春节、清明、五一" />
          </Form.Item>
          <Form.Item name="start_date" label="开始日期" rules={[{ required: true }]}>
            <DatePicker style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="end_date" label="结束日期" rules={[{ required: true }]}>
            <DatePicker style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="type" label="类型">
            <Select options={[{ value: '法定', label: '法定' }, { value: '调休', label: '调休' }]} />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  )
}
